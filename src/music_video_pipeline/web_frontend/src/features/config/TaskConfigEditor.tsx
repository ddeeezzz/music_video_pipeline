import { useMemo } from "react";

import { Collapse, Descriptions, Input, Typography } from "antd";

/** 需要暴露给用户的 section 键名（其余全部由 config.py 默认值控制，不暴露给前端） */
const VISIBLE_SECTIONS = new Set(["paths"]);

const SECTION_LABELS: Record<string, string> = {
  paths: "路径",
};

/** 字段中不应展示的敏感键名 */
const SENSITIVE_KEYS = new Set(["api_key_file", "acoustid_api_key_file"]);

interface TaskConfigEditorProps {
  config: Record<string, unknown>;
  /** 当前覆盖值（editable 模式下使用） */
  overrides?: Record<string, unknown>;
  /** 是否只读 */
  readOnly?: boolean;
  /** 值变更回调 (section, key, value) */
  onChange?: (section: string, key: string, value: unknown) => void;
}

/** 判断值是否为简单类型（可安全展示/编辑） */
function isSimpleValue(value: unknown): boolean {
  return (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean" ||
    value === null ||
    value === undefined
  );
}

/** 判断值是否为简单数组（纯字符串/数字/布尔） */
function isSimpleArray(value: unknown): boolean {
  return Array.isArray(value) && value.every((v) => isSimpleValue(v));
}

/** 将简单值转为展示文本 */
function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "-";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

/**
 * TaskConfigEditor
 * 按 section 折叠展示/编辑配置对象。
 * 只读模式：Descriptions 组件展示键值对。
 * 编辑模式：Input 组件允许修改。
 */
export function TaskConfigEditor({ config, overrides, readOnly = true, onChange }: TaskConfigEditorProps) {
  const sections = useMemo(() => {
    const entries = Object.entries(config).filter(([sectionKey]) => VISIBLE_SECTIONS.has(sectionKey));
    return entries.map(([sectionKey, sectionValue]) => {
      const label = SECTION_LABELS[sectionKey] || sectionKey;
      const sectionOverrides = (overrides?.[sectionKey] as Record<string, unknown>) || {};
      let items = extractSimpleFields(sectionValue as Record<string, unknown>, sectionOverrides);
      // 从 module_b 中提取故事模板路径，追加到 paths 分段
      if (sectionKey === "paths") {
        const moduleBConfig = config.module_b as Record<string, unknown> | undefined;
        const moduleBOverrides = overrides?.module_b as Record<string, unknown> | undefined;
        const tmplVal = moduleBOverrides?.storyboard_template_file ?? moduleBConfig?.storyboard_template_file;
        if (tmplVal !== undefined) {
          items.push({
            fieldKey: "storyboard_template_file",
            displayValue: formatValue(tmplVal),
            rawValue: tmplVal,
            isOverridden: moduleBOverrides?.storyboard_template_file !== undefined,
            targetSection: "module_b",
          });
        }
      }
      return { key: sectionKey, label, items };
    });
  }, [config, overrides]);

  // 用 config 对象序列化长度做版本标记，数据变化时 Input 重新初始化
  const configVersion = useMemo(() => {
    try { return JSON.stringify(config).length } catch { return 0 }
  }, [config]);

  return (
    <Collapse
      items={sections.map(({ key, label, items }) => ({
        key,
        label: (
          <Typography.Text strong>
            {label}
            <Typography.Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
              ({items.length} 项)
            </Typography.Text>
          </Typography.Text>
        ),
        children: readOnly ? (
          <Descriptions column={2} size="small" bordered>
            {items.map(({ fieldKey, displayValue, isOverridden }) => (
              <Descriptions.Item
                key={fieldKey}
                label={
                  <span style={isOverridden ? { color: "#1677ff", fontWeight: 600 } : undefined}>
                    {fieldKey}
                    {isOverridden ? " *" : ""}
                  </span>
                }
              >
                {displayValue}
              </Descriptions.Item>
            ))}
          </Descriptions>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {items.map(({ fieldKey, displayValue, rawValue, isOverridden, targetSection }) => {
              if (SENSITIVE_KEYS.has(fieldKey)) return null;
              return (
                <div key={fieldKey} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Typography.Text
                    style={{ minWidth: 180, flexShrink: 0 }}
                    type={isOverridden ? undefined : "secondary"}
                  >
                    {fieldKey}
                  </Typography.Text>
                  <Input
                    size="small"
                    key={`${configVersion}-${key}-${fieldKey}`}
                    defaultValue={String(rawValue ?? "")}
                    style={{ flex: 1, maxWidth: 400 }}
                    onBlur={(e) => {
                      const val = e.target.value;
                      const original = String(rawValue ?? "");
                      // 只有值真正变化时才触发 onChange
                      if (val !== original) {
                        // _targetSection 用于跨 section 的字段（如 storyboard_template_file）
                        onChange?.(targetSection || key, fieldKey, convertValue(rawValue, val));
                      }
                    }}
                  />
                  {isOverridden && (
                    <Typography.Text type="warning" style={{ fontSize: 12 }}>
                      已修改
                    </Typography.Text>
                  )}
                </div>
              );
            })}
          </div>
        ),
      }))}
    />
  );
}

/** 从 section 数据中提取可展示的扁平字段 */
type ConfigFieldItem = {
  fieldKey: string;
  displayValue: string;
  rawValue: unknown;
  isOverridden: boolean;
  targetSection?: string;
};

function extractSimpleFields(
  sectionData: Record<string, unknown>,
  sectionOverrides: Record<string, unknown>,
): ConfigFieldItem[] {
  const result: ConfigFieldItem[] = [];
  for (const [key, value] of Object.entries(sectionData)) {
    if (SENSITIVE_KEYS.has(key)) continue;
    if (value !== null && typeof value === "object" && !isSimpleArray(value)) {
      // 嵌套对象递归展平，用点分隔
      const nested = extractSimpleFields(value as Record<string, unknown>, {});
      for (const item of nested) {
        result.push({
          fieldKey: `${key}.${item.fieldKey}`,
          displayValue: item.displayValue,
          rawValue: item.rawValue,
          isOverridden: item.isOverridden || key in sectionOverrides,
        });
      }
    } else {
      const overrideVal = sectionOverrides[key];
      const isOverridden = overrideVal !== undefined;
      result.push({
        fieldKey: key,
        displayValue: formatValue(isOverridden ? overrideVal : value),
        rawValue: isOverridden ? overrideVal : value,
        isOverridden,
      });
    }
  }
  result.sort((a, b) => a.fieldKey.localeCompare(b.fieldKey));
  return result;
}

/**
 * 根据原始值类型转换输入字符串。
 * 数字 → parseFloat，布尔 → "true"/"false" 解析，字符串保持原样。
 */
function convertValue(original: unknown, input: string): unknown {
  if (typeof original === "number") {
    const parsed = parseFloat(input);
    return Number.isNaN(parsed) ? input : parsed;
  }
  if (typeof original === "boolean") {
    if (input === "true" || input === "1") return true;
    if (input === "false" || input === "0") return false;
    return input;
  }
  return input;
}
