"""
文件用途：提供 QQ 音乐 QRC 密文的最小解密与文本抽取能力。
核心流程：先对 hex 密文执行 3DES 解密和 zlib 解压，再从 QRC XML 中提取原文行。
输入输出：输入 QQ musicu 返回的 lyric/trans/roma 密文字段，输出可展示的行文本或 LRC 文本。
依赖说明：仅依赖标准库，不引入额外第三方密码学包。
维护说明：本文件刻意只保留当前项目接入 QQ 罗马音所需的最小子集。
"""

from __future__ import annotations

import html
import re
from zlib import decompress


QQ_MUSIC_QRC_KEY = b"!@#)(*$%123ZXC!@!@#)(NHL"
QQ_MUSIC_DECRYPT = 0

_QRC_XML_PATTERN = re.compile(r'<Lyric_1 LyricType="1" LyricContent="(?P<content>.*?)"/>', re.DOTALL)
_QRC_LINE_PATTERN = re.compile(r"^\[(?P<start>\d+),(?P<duration>\d+)\](?P<content>.*)$")
_QRC_WORD_PATTERN = re.compile(r"(?:\[\d+,\d+\])?(?P<content>(?:(?!\(\d+,\d+\)).)*)\((?P<start>\d+),(?P<duration>\d+)\)")
_QRC_EMPTY_LINE_PATTERN = re.compile(r"^\(\d+,\d+\)$")


def decrypt_qq_music_qrc(encrypted_qrc_hex: str) -> str:
    """
    功能说明：解密 QQ musicu 返回的 QRC 十六进制密文。
    参数说明：
    - encrypted_qrc_hex: 十六进制密文字符串。
    返回值：
    - str: 解密后的 UTF-8 QRC 文本。
    异常说明：输入为空或解密失败时抛出 ValueError。
    边界条件：按 8 字节分块执行 3DES，再做 zlib 解压。
    """
    normalized_hex = str(encrypted_qrc_hex).strip()
    if not normalized_hex:
        raise ValueError("encrypted_qrc_hex is empty")
    try:
        encrypted_bytes = bytearray.fromhex(normalized_hex)
        key_schedule = _tripledes_key_setup(QQ_MUSIC_QRC_KEY, QQ_MUSIC_DECRYPT)
        decrypted_bytes = bytearray()
        for index in range(0, len(encrypted_bytes), 8):
            decrypted_bytes.extend(_tripledes_crypt(encrypted_bytes[index : index + 8], key_schedule))
        return decompress(decrypted_bytes).decode("utf-8")
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"decrypt qq music qrc failed: {error}") from error


def extract_plaintext_from_qq_music_qrc(qrc_text: str) -> str:
    """
    功能说明：从解密后的 QRC 文本中提取纯文本歌词行。
    参数说明：
    - qrc_text: 解密后的 QRC 文本。
    返回值：
    - str: 逐行纯文本歌词。
    异常说明：格式不合法时返回空字符串。
    边界条件：保留原始行顺序，忽略 metadata tag 与空时间占位行。
    """
    lyric_content = _extract_qrc_lyric_content(qrc_text=qrc_text)
    if not lyric_content:
        return ""
    plain_lines: list[str] = []
    for raw_line in lyric_content.splitlines():
        normalized_line = raw_line.strip()
        line_match = _QRC_LINE_PATTERN.match(normalized_line)
        if line_match is None:
            continue
        line_content = str(line_match.group("content") or "").strip()
        if not line_content or _QRC_EMPTY_LINE_PATTERN.match(line_content):
            continue
        plain_text = _extract_qrc_line_text(line_content=line_content).strip()
        if plain_text:
            plain_lines.append(plain_text)
    return "\n".join(plain_lines).strip()


def extract_lrc_from_qq_music_qrc(qrc_text: str) -> str:
    """
    功能说明：从解密后的 QRC 文本中提取行级 LRC 文本。
    参数说明：
    - qrc_text: 解密后的 QRC 文本。
    返回值：
    - str: 含行级时间戳的 LRC 文本。
    异常说明：格式不合法时返回空字符串。
    边界条件：仅输出行级时间戳，不保留词级时间信息。
    """
    lyric_content = _extract_qrc_lyric_content(qrc_text=qrc_text)
    if not lyric_content:
        return ""
    lrc_lines: list[str] = []
    for raw_line in lyric_content.splitlines():
        normalized_line = raw_line.strip()
        line_match = _QRC_LINE_PATTERN.match(normalized_line)
        if line_match is None:
            continue
        line_start_ms = int(line_match.group("start") or 0)
        line_content = str(line_match.group("content") or "").strip()
        if not line_content or _QRC_EMPTY_LINE_PATTERN.match(line_content):
            continue
        line_text = _extract_qrc_line_text(line_content=line_content).strip()
        if line_text:
            lrc_lines.append(f"[{_format_lrc_timestamp(line_start_ms)}]{line_text}")
    return "\n".join(lrc_lines).strip()


def extract_enhanced_lrc_from_qq_music_qrc(qrc_text: str) -> str:
    """
    功能说明：从解密后的 QRC 文本中提取增强 LRC 文本。
    参数说明：
    - qrc_text: 解密后的 QRC 文本。
    返回值：
    - str: 含行级时间戳与词级 `<start>词<end>` 标记的增强 LRC。
    异常说明：格式不合法时返回空字符串。
    边界条件：没有词级标记时回退为普通行级 LRC。
    """
    lyric_content = _extract_qrc_lyric_content(qrc_text=qrc_text)
    if not lyric_content:
        return ""
    enhanced_lines: list[str] = []
    for raw_line in lyric_content.splitlines():
        normalized_line = raw_line.strip()
        line_match = _QRC_LINE_PATTERN.match(normalized_line)
        if line_match is None:
            continue
        line_start_ms = int(line_match.group("start") or 0)
        line_content = str(line_match.group("content") or "").strip()
        if not line_content or _QRC_EMPTY_LINE_PATTERN.match(line_content):
            continue
        words = _extract_qrc_word_units(line_content=line_content, line_start_ms=line_start_ms)
        if words:
            enhanced_lines.append(
                f"[{_format_lrc_timestamp(line_start_ms)}]"
                + "".join(
                    f"<{_format_lrc_timestamp(int(word['start_ms']))}>{str(word['text'])}<"
                    f"{_format_lrc_timestamp(int(word['end_ms']))}>"
                    for word in words
                    if str(word.get("text", ""))
                )
            )
            continue
        line_text = _extract_qrc_line_text(line_content=line_content).strip()
        if line_text:
            enhanced_lines.append(f"[{_format_lrc_timestamp(line_start_ms)}]{line_text}")
    return "\n".join(enhanced_lines).strip()


def _extract_qrc_lyric_content(qrc_text: str) -> str:
    """
    功能说明：提取 QRC XML 中的 LyricContent 属性文本。
    参数说明：
    - qrc_text: 解密后的 QRC 文本。
    返回值：
    - str: HTML 反转义后的歌词主体文本。
    异常说明：无。
    边界条件：匹配失败时返回空字符串。
    """
    qrc_match = _QRC_XML_PATTERN.search(str(qrc_text or ""))
    if qrc_match is None:
        return ""
    return html.unescape(str(qrc_match.group("content") or "")).strip()


def _extract_qrc_line_text(line_content: str) -> str:
    """
    功能说明：从单行 QRC 内容中移除词级时间标记并拼出展示文本。
    参数说明：
    - line_content: 去掉行级时间后的一整行内容。
    返回值：
    - str: 单行纯文本。
    异常说明：无。
    边界条件：没有词级标记时回退为简单移除残留时间片段。
    """
    words = [
        str(word_match.group("content") or "")
        for word_match in _QRC_WORD_PATTERN.finditer(str(line_content or ""))
        if str(word_match.group("content") or "") != "\r"
    ]
    if words:
        return "".join(words)
    return re.sub(r"\(\d+,\d+\)", "", str(line_content or "")).strip()


def _extract_qrc_word_units(line_content: str, line_start_ms: int) -> list[dict[str, int | str]]:
    """
    功能说明：从单行 QRC 内容中提取词级时间与文本。
    参数说明：
    - line_content: 去掉行级时间后的单行内容。
    - line_start_ms: 行起始时间毫秒值。
    返回值：
    - list[dict[str, int | str]]: 词级时间单元列表。
    异常说明：无。
    边界条件：会忽略空文本与仅回车占位的 token。
    """
    word_units: list[dict[str, int | str]] = []
    for word_match in _QRC_WORD_PATTERN.finditer(str(line_content or "")):
        word_text = str(word_match.group("content") or "")
        if word_text == "\r":
            continue
        normalized_word_text = word_text.strip()
        if not normalized_word_text:
            continue
        word_start_ms = int(line_start_ms) + int(word_match.group("start") or 0)
        word_end_ms = word_start_ms + int(word_match.group("duration") or 0)
        word_units.append(
            {
                "text": normalized_word_text,
                "start_ms": word_start_ms,
                "end_ms": word_end_ms,
            }
        )
    return word_units


def _format_lrc_timestamp(milliseconds: int) -> str:
    """
    功能说明：把毫秒时间格式化为 LRC 行级时间戳。
    参数说明：
    - milliseconds: 时间毫秒值。
    返回值：
    - str: `MM:SS.xx` 形式的时间戳。
    异常说明：无。
    边界条件：向下保留到 10ms 精度，满足当前解析链需要。
    """
    safe_milliseconds = max(0, int(milliseconds))
    total_centiseconds = safe_milliseconds // 10
    minutes = total_centiseconds // 6000
    seconds = (total_centiseconds % 6000) // 100
    centiseconds = total_centiseconds % 100
    return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _bitnum(data: bytearray | bytes, index: int, shift: int) -> int:
    return ((data[(index // 32) * 4 + 3 - (index % 32) // 8] >> (7 - index % 8)) & 1) << shift


def _bitnum_intr(value: int, index: int, shift: int) -> int:
    return ((value >> (31 - index)) & 1) << shift


def _bitnum_intl(value: int, index: int, shift: int) -> int:
    return ((value << index) & 0x80000000) >> shift


def _sbox_bit(value: int) -> int:
    return (value & 32) | ((value & 31) >> 1) | ((value & 1) << 4)


_SBOX = (
    (14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7, 0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8, 4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0, 15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13),
    (15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10, 3, 13, 4, 7, 15, 2, 8, 15, 12, 0, 1, 10, 6, 9, 11, 5, 0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15, 13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9),
    (10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8, 13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1, 13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7, 1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12),
    (7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15, 13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9, 10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4, 3, 15, 0, 6, 10, 10, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14),
    (2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9, 14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6, 4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14, 11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3),
    (12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11, 10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8, 9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6, 4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13),
    (4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1, 13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6, 1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2, 6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12),
    (13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7, 1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2, 7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8, 2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11),
)


def _initial_permutation(input_data: bytearray) -> tuple[int, int]:
    return (
        (
            _bitnum(input_data, 57, 31) | _bitnum(input_data, 49, 30) | _bitnum(input_data, 41, 29) | _bitnum(input_data, 33, 28)
            | _bitnum(input_data, 25, 27) | _bitnum(input_data, 17, 26) | _bitnum(input_data, 9, 25) | _bitnum(input_data, 1, 24)
            | _bitnum(input_data, 59, 23) | _bitnum(input_data, 51, 22) | _bitnum(input_data, 43, 21) | _bitnum(input_data, 35, 20)
            | _bitnum(input_data, 27, 19) | _bitnum(input_data, 19, 18) | _bitnum(input_data, 11, 17) | _bitnum(input_data, 3, 16)
            | _bitnum(input_data, 61, 15) | _bitnum(input_data, 53, 14) | _bitnum(input_data, 45, 13) | _bitnum(input_data, 37, 12)
            | _bitnum(input_data, 29, 11) | _bitnum(input_data, 21, 10) | _bitnum(input_data, 13, 9) | _bitnum(input_data, 5, 8)
            | _bitnum(input_data, 63, 7) | _bitnum(input_data, 55, 6) | _bitnum(input_data, 47, 5) | _bitnum(input_data, 39, 4)
            | _bitnum(input_data, 31, 3) | _bitnum(input_data, 23, 2) | _bitnum(input_data, 15, 1) | _bitnum(input_data, 7, 0)
        ),
        (
            _bitnum(input_data, 56, 31) | _bitnum(input_data, 48, 30) | _bitnum(input_data, 40, 29) | _bitnum(input_data, 32, 28)
            | _bitnum(input_data, 24, 27) | _bitnum(input_data, 16, 26) | _bitnum(input_data, 8, 25) | _bitnum(input_data, 0, 24)
            | _bitnum(input_data, 58, 23) | _bitnum(input_data, 50, 22) | _bitnum(input_data, 42, 21) | _bitnum(input_data, 34, 20)
            | _bitnum(input_data, 26, 19) | _bitnum(input_data, 18, 18) | _bitnum(input_data, 10, 17) | _bitnum(input_data, 2, 16)
            | _bitnum(input_data, 60, 15) | _bitnum(input_data, 52, 14) | _bitnum(input_data, 44, 13) | _bitnum(input_data, 36, 12)
            | _bitnum(input_data, 28, 11) | _bitnum(input_data, 20, 10) | _bitnum(input_data, 12, 9) | _bitnum(input_data, 4, 8)
            | _bitnum(input_data, 62, 7) | _bitnum(input_data, 54, 6) | _bitnum(input_data, 46, 5) | _bitnum(input_data, 38, 4)
            | _bitnum(input_data, 30, 3) | _bitnum(input_data, 22, 2) | _bitnum(input_data, 14, 1) | _bitnum(input_data, 6, 0)
        ),
    )


def _inverse_permutation(left: int, right: int) -> bytearray:
    output = bytearray(8)
    output[3] = _bitnum_intr(right, 7, 7) | _bitnum_intr(left, 7, 6) | _bitnum_intr(right, 15, 5) | _bitnum_intr(left, 15, 4) | _bitnum_intr(right, 23, 3) | _bitnum_intr(left, 23, 2) | _bitnum_intr(right, 31, 1) | _bitnum_intr(left, 31, 0)
    output[2] = _bitnum_intr(right, 6, 7) | _bitnum_intr(left, 6, 6) | _bitnum_intr(right, 14, 5) | _bitnum_intr(left, 14, 4) | _bitnum_intr(right, 22, 3) | _bitnum_intr(left, 22, 2) | _bitnum_intr(right, 30, 1) | _bitnum_intr(left, 30, 0)
    output[1] = _bitnum_intr(right, 5, 7) | _bitnum_intr(left, 5, 6) | _bitnum_intr(right, 13, 5) | _bitnum_intr(left, 13, 4) | _bitnum_intr(right, 21, 3) | _bitnum_intr(left, 21, 2) | _bitnum_intr(right, 29, 1) | _bitnum_intr(left, 29, 0)
    output[0] = _bitnum_intr(right, 4, 7) | _bitnum_intr(left, 4, 6) | _bitnum_intr(right, 12, 5) | _bitnum_intr(left, 12, 4) | _bitnum_intr(right, 20, 3) | _bitnum_intr(left, 20, 2) | _bitnum_intr(right, 28, 1) | _bitnum_intr(left, 28, 0)
    output[7] = _bitnum_intr(right, 3, 7) | _bitnum_intr(left, 3, 6) | _bitnum_intr(right, 11, 5) | _bitnum_intr(left, 11, 4) | _bitnum_intr(right, 19, 3) | _bitnum_intr(left, 19, 2) | _bitnum_intr(right, 27, 1) | _bitnum_intr(left, 27, 0)
    output[6] = _bitnum_intr(right, 2, 7) | _bitnum_intr(left, 2, 6) | _bitnum_intr(right, 10, 5) | _bitnum_intr(left, 10, 4) | _bitnum_intr(right, 18, 3) | _bitnum_intr(left, 18, 2) | _bitnum_intr(right, 26, 1) | _bitnum_intr(left, 26, 0)
    output[5] = _bitnum_intr(right, 1, 7) | _bitnum_intr(left, 1, 6) | _bitnum_intr(right, 9, 5) | _bitnum_intr(left, 9, 4) | _bitnum_intr(right, 17, 3) | _bitnum_intr(left, 17, 2) | _bitnum_intr(right, 25, 1) | _bitnum_intr(left, 25, 0)
    output[4] = _bitnum_intr(right, 0, 7) | _bitnum_intr(left, 0, 6) | _bitnum_intr(right, 8, 5) | _bitnum_intr(left, 8, 4) | _bitnum_intr(right, 16, 3) | _bitnum_intr(left, 16, 2) | _bitnum_intr(right, 24, 1) | _bitnum_intr(left, 24, 0)
    return output


def _feistel(state: int, key: list[int]) -> int:
    temp_left = (
        _bitnum_intl(state, 31, 0) | ((state & 0xF0000000) >> 1) | _bitnum_intl(state, 4, 5) | _bitnum_intl(state, 3, 6)
        | ((state & 0x0F000000) >> 3) | _bitnum_intl(state, 8, 11) | _bitnum_intl(state, 7, 12) | ((state & 0x00F00000) >> 5)
        | _bitnum_intl(state, 12, 17) | _bitnum_intl(state, 11, 18) | ((state & 0x000F0000) >> 7) | _bitnum_intl(state, 16, 23)
    )
    temp_right = (
        _bitnum_intl(state, 15, 0) | ((state & 0x0000F000) << 15) | _bitnum_intl(state, 20, 5) | _bitnum_intl(state, 19, 6)
        | ((state & 0x00000F00) << 13) | _bitnum_intl(state, 24, 11) | _bitnum_intl(state, 23, 12) | ((state & 0x000000F0) << 11)
        | _bitnum_intl(state, 28, 17) | _bitnum_intl(state, 27, 18) | ((state & 0x0000000F) << 9) | _bitnum_intl(state, 0, 23)
    )
    expanded = [
        (temp_left >> 24) & 0xFF,
        (temp_left >> 16) & 0xFF,
        (temp_left >> 8) & 0xFF,
        (temp_right >> 24) & 0xFF,
        (temp_right >> 16) & 0xFF,
        (temp_right >> 8) & 0xFF,
    ]
    mixed = [expanded[index] ^ key[index] for index in range(6)]
    substituted = (
        (_SBOX[0][_sbox_bit(mixed[0] >> 2)] << 28)
        | (_SBOX[1][_sbox_bit(((mixed[0] & 0x03) << 4) | (mixed[1] >> 4))] << 24)
        | (_SBOX[2][_sbox_bit(((mixed[1] & 0x0F) << 2) | (mixed[2] >> 6))] << 20)
        | (_SBOX[3][_sbox_bit(mixed[2] & 0x3F)] << 16)
        | (_SBOX[4][_sbox_bit(mixed[3] >> 2)] << 12)
        | (_SBOX[5][_sbox_bit(((mixed[3] & 0x03) << 4) | (mixed[4] >> 4))] << 8)
        | (_SBOX[6][_sbox_bit(((mixed[4] & 0x0F) << 2) | (mixed[5] >> 6))] << 4)
        | _SBOX[7][_sbox_bit(mixed[5] & 0x3F)]
    )
    return (
        _bitnum_intl(substituted, 15, 0) | _bitnum_intl(substituted, 6, 1) | _bitnum_intl(substituted, 19, 2) | _bitnum_intl(substituted, 20, 3)
        | _bitnum_intl(substituted, 28, 4) | _bitnum_intl(substituted, 11, 5) | _bitnum_intl(substituted, 27, 6) | _bitnum_intl(substituted, 16, 7)
        | _bitnum_intl(substituted, 0, 8) | _bitnum_intl(substituted, 14, 9) | _bitnum_intl(substituted, 22, 10) | _bitnum_intl(substituted, 25, 11)
        | _bitnum_intl(substituted, 4, 12) | _bitnum_intl(substituted, 17, 13) | _bitnum_intl(substituted, 30, 14) | _bitnum_intl(substituted, 9, 15)
        | _bitnum_intl(substituted, 1, 16) | _bitnum_intl(substituted, 7, 17) | _bitnum_intl(substituted, 23, 18) | _bitnum_intl(substituted, 13, 19)
        | _bitnum_intl(substituted, 31, 20) | _bitnum_intl(substituted, 26, 21) | _bitnum_intl(substituted, 2, 22) | _bitnum_intl(substituted, 8, 23)
        | _bitnum_intl(substituted, 18, 24) | _bitnum_intl(substituted, 12, 25) | _bitnum_intl(substituted, 29, 26) | _bitnum_intl(substituted, 5, 27)
        | _bitnum_intl(substituted, 21, 28) | _bitnum_intl(substituted, 10, 29) | _bitnum_intl(substituted, 3, 30) | _bitnum_intl(substituted, 24, 31)
    )


def _des_crypt(block: bytearray, key_schedule: list[list[int]]) -> bytearray:
    left, right = _initial_permutation(block)
    for round_index in range(15):
        previous_right = right
        right = _feistel(right, key_schedule[round_index]) ^ left
        left = previous_right
    left = _feistel(right, key_schedule[15]) ^ left
    return _inverse_permutation(left, right)


def _des_key_schedule(key: bytes, mode: int) -> list[list[int]]:
    schedule = [[0] * 6 for _ in range(16)]
    round_shift = (1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1)
    perm_c = (56, 48, 40, 32, 24, 16, 8, 0, 57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18, 10, 2, 59, 51, 43, 35)
    perm_d = (62, 54, 46, 38, 30, 22, 14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 60, 52, 44, 36, 28, 20, 12, 4, 27, 19, 11, 3)
    compression = (13, 16, 10, 23, 0, 4, 2, 27, 14, 5, 20, 9, 22, 18, 11, 3, 25, 7, 15, 6, 26, 19, 12, 1, 40, 51, 30, 36, 46, 54, 29, 39, 50, 44, 32, 47, 43, 48, 38, 55, 33, 52, 45, 41, 49, 35, 28, 31)
    c_value = sum(_bitnum(key, perm_c[index], 31 - index) for index in range(28))
    d_value = sum(_bitnum(key, perm_d[index], 31 - index) for index in range(28))
    for round_index in range(16):
        c_value = ((c_value << round_shift[round_index]) | (c_value >> (28 - round_shift[round_index]))) & 0xFFFFFFF0
        d_value = ((d_value << round_shift[round_index]) | (d_value >> (28 - round_shift[round_index]))) & 0xFFFFFFF0
        target_index = 15 - round_index if mode == QQ_MUSIC_DECRYPT else round_index
        for byte_index in range(24):
            schedule[target_index][byte_index // 8] |= _bitnum_intr(c_value, compression[byte_index], 7 - (byte_index % 8))
        for byte_index in range(24, 48):
            schedule[target_index][byte_index // 8] |= _bitnum_intr(d_value, compression[byte_index] - 27, 7 - (byte_index % 8))
    return schedule


def _tripledes_key_setup(key: bytes, mode: int) -> list[list[list[int]]]:
    return [
        _des_key_schedule(key[16:], QQ_MUSIC_DECRYPT),
        _des_key_schedule(key[8:], 1),
        _des_key_schedule(key[0:], QQ_MUSIC_DECRYPT),
    ] if mode == QQ_MUSIC_DECRYPT else [
        _des_key_schedule(key[0:], 1),
        _des_key_schedule(key[8:], QQ_MUSIC_DECRYPT),
        _des_key_schedule(key[16:], 1),
    ]


def _tripledes_crypt(block: bytearray, key_schedule: list[list[list[int]]]) -> bytearray:
    working_block = bytearray(block)
    if len(working_block) < 8:
        working_block.extend(b"\x00" * (8 - len(working_block)))
    for stage_key in key_schedule:
        working_block = _des_crypt(working_block, stage_key)
    return working_block
