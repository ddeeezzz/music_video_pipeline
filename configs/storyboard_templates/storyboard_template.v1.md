# Storyboard Template v1

本文件是模块 B v2 的正式编排模板源。机器解析固定的 `## section` 与 `### subsection` 结构，不再读取 JSON code fence。

## template_meta

- template_id: storyboard_template_v1_monochrome_cat_hide_seek

## style

- color_mode: 黑白
- render_style: 日本漫画风插图，简约的背景

## story

- premise_zh: 黑猫与少女在空无一人的城市空间里进行带有不安感的捉迷藏。

## scene_catalog

### scene_alley_dim
- name_zh: 昏暗小巷
- description_zh: 昏暗的小巷

### scene_corridor_uniform
- name_zh: 均匀走廊
- description_zh: 压抑、房门排布的走廊

### scene_playground_basketball
- name_zh: 学校操场
- description_zh: 篮球操场，只有篮球架，没有人，没有篮球

### scene_kitchen_messy
- name_zh: 小餐馆后厨
- description_zh: 杂乱、狭窄的小餐馆后厨

## prop_catalog

### prop_cage_wire
- name_zh: 铁丝笼子
- description_zh: 旧铁丝笼，带压迫感

### prop_rope_fiber
- name_zh: 纤维绳
- description_zh: 粗糙纤维绳，打结

## character_catalog

### character_black_cat
- name_zh: 黑猫
- description_zh: 瘦长、警觉的黑猫

### character_faceless_girl
- name_zh: 少女
- description_zh: 水手服少女，黑长直

## composition_catalog

### comp_sym_center
- name_zh: 对称中轴
- description_zh: 主体置于中轴，左右均衡
- prompt_tags_en: symmetrical composition, center framing, balanced negative space
- safe_for_closeup: true
- safe_for_motion: true

### comp_left_third_profile
- name_zh: 左三分侧置
- description_zh: 主体压在左三分线上，留出右侧空间
- prompt_tags_en: rule of thirds, subject on left third, profile staging
- safe_for_closeup: true
- safe_for_motion: true

### comp_frame_within_frame
- name_zh: 框中框
- description_zh: 前景元素框定主体
- prompt_tags_en: frame within frame, doorway framing, voyeuristic composition
- safe_for_closeup: true
- safe_for_motion: false

## camera_plan_presets

### none
- mode: none
- direction: center
- strength: none
- easing: linear

### zoom_in_s
- mode: zoom
- direction: center
- strength: small
- easing: ease_in_out

### pan_left_s
- mode: pan
- direction: left
- strength: small
- easing: linear

### pan_right_s
- mode: pan
- direction: right
- strength: small
- easing: linear

## transition_presets

### none
- kind: none
- duration_ms: 0
- easing: linear

### hard_cut_0
- kind: hard_cut
- duration_ms: 0
- easing: linear

### crossfade_160
- kind: crossfade
- duration_ms: 160
- easing: ease_in_out

### crossfade_240
- kind: crossfade
- duration_ms: 240
- easing: ease_in_out

### fade_black_240
- kind: fade_black
- duration_ms: 240
- easing: ease_in_out

### fade_white_200
- kind: fade_white
- duration_ms: 200
- easing: ease_in_out

### wipe_left_200
- kind: wipe_left
- duration_ms: 200
- easing: ease_in_out

### wipe_right_200
- kind: wipe_right
- duration_ms: 200
- easing: ease_in_out
