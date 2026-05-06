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

### comp_right_third_profile
- name_zh: 右三分侧置
- description_zh: 主体压在右三分线上，留出左侧空间
- prompt_tags_en: rule of thirds, subject on right third, profile staging
- safe_for_closeup: true
- safe_for_motion: true

### comp_vanishing_corridor
- name_zh: 消失点走廊
- description_zh: 强透视纵深，指向消失点
- prompt_tags_en: vanishing point, deep corridor perspective, tunnel framing
- safe_for_closeup: false
- safe_for_motion: true

### comp_frame_within_frame
- name_zh: 框中框
- description_zh: 前景元素框定主体
- prompt_tags_en: frame within frame, doorway framing, voyeuristic composition
- safe_for_closeup: true
- safe_for_motion: false

### comp_negative_space_left
- name_zh: 左留白压迫
- description_zh: 主体偏右，大面积左侧留白
- prompt_tags_en: negative space left, off-center subject, asymmetric tension
- safe_for_closeup: false
- safe_for_motion: true

### comp_negative_space_right
- name_zh: 右留白压迫
- description_zh: 主体偏左，大面积右侧留白
- prompt_tags_en: negative space right, off-center subject, asymmetric tension
- safe_for_closeup: false
- safe_for_motion: true

### comp_high_angle_isolation
- name_zh: 高角度孤立
- description_zh: 较高俯视角，突出角色的渺小
- prompt_tags_en: high angle, isolated subject, vulnerable staging
- safe_for_closeup: false
- safe_for_motion: false

### comp_low_angle_presence
- name_zh: 低角度存在感
- description_zh: 较低仰视角，抬高主体
- prompt_tags_en: low angle, strong presence, looming subject
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

### zoom_out_s
- mode: zoom
- direction: center
- strength: small
- easing: ease_in_out

### zoom_in_m
- mode: zoom
- direction: center
- strength: medium
- easing: ease_in_out

### zoom_out_m
- mode: zoom
- direction: center
- strength: medium
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

### pan_up_s
- mode: pan
- direction: up
- strength: small
- easing: linear

### pan_down_s
- mode: pan
- direction: down
- strength: small
- easing: linear

### pan_ul_s
- mode: pan
- direction: up_left
- strength: small
- easing: ease_in_out

### pan_ur_s
- mode: pan
- direction: up_right
- strength: small
- easing: ease_in_out

### pan_dl_s
- mode: pan
- direction: down_left
- strength: small
- easing: ease_in_out

### pan_dr_s
- mode: pan
- direction: down_right
- strength: small
- easing: ease_in_out

## camera_mapping

### low_down
- energy_level: low
- trend: down
- default_preset_id: none
- candidate_preset_ids: none, zoom_out_s, pan_left_s

### low_flat
- energy_level: low
- trend: flat
- default_preset_id: none
- candidate_preset_ids: none, zoom_in_s, pan_right_s

### low_up
- energy_level: low
- trend: up
- default_preset_id: zoom_in_s
- candidate_preset_ids: none, zoom_in_s, pan_ur_s

### mid_down
- energy_level: mid
- trend: down
- default_preset_id: pan_left_s
- candidate_preset_ids: none, pan_left_s, zoom_out_s

### mid_flat
- energy_level: mid
- trend: flat
- default_preset_id: pan_right_s
- candidate_preset_ids: none, pan_right_s, zoom_in_s

### mid_up
- energy_level: mid
- trend: up
- default_preset_id: zoom_in_s
- candidate_preset_ids: none, zoom_in_s, pan_ur_s

### high_down
- energy_level: high
- trend: down
- default_preset_id: pan_dl_s
- candidate_preset_ids: none, pan_dl_s, zoom_out_m

### high_flat
- energy_level: high
- trend: flat
- default_preset_id: zoom_in_m
- candidate_preset_ids: none, zoom_in_m, pan_right_s

### high_up
- energy_level: high
- trend: up
- default_preset_id: pan_ur_s
- candidate_preset_ids: none, pan_ur_s, zoom_in_m

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

## transition_mapping

### low_low
- current_energy_level: low
- next_energy_level: low
- default_preset_id: none
- candidate_preset_ids: none, crossfade_160

### low_mid
- current_energy_level: low
- next_energy_level: mid
- default_preset_id: crossfade_160
- candidate_preset_ids: none, crossfade_160, wipe_left_200

### low_high
- current_energy_level: low
- next_energy_level: high
- default_preset_id: fade_white_200
- candidate_preset_ids: none, fade_white_200, wipe_left_200

### mid_low
- current_energy_level: mid
- next_energy_level: low
- default_preset_id: crossfade_160
- candidate_preset_ids: none, crossfade_160, fade_black_240

### mid_mid
- current_energy_level: mid
- next_energy_level: mid
- default_preset_id: crossfade_240
- candidate_preset_ids: none, crossfade_240, hard_cut_0

### mid_high
- current_energy_level: mid
- next_energy_level: high
- default_preset_id: wipe_left_200
- candidate_preset_ids: none, wipe_left_200, hard_cut_0

### high_low
- current_energy_level: high
- next_energy_level: low
- default_preset_id: fade_black_240
- candidate_preset_ids: none, fade_black_240, crossfade_160

### high_mid
- current_energy_level: high
- next_energy_level: mid
- default_preset_id: hard_cut_0
- candidate_preset_ids: none, hard_cut_0, wipe_right_200

### high_high
- current_energy_level: high
- next_energy_level: high
- default_preset_id: hard_cut_0
- candidate_preset_ids: none, hard_cut_0, wipe_right_200
