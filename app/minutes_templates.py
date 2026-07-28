from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from .domain import utc_now


DEFAULT_TEMPLATE_ID = "lab-meeting"
_SAFE_TEMPLATE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SAFE_SECTION_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_VALID_SECTION_KINDS = {"summary", "list", "actions"}
MAX_CUSTOM_SECTIONS = 8

CORE_SECTION_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "key": "summary",
        "title": "会议摘要",
        "kind": "summary",
        "required": True,
        "guidance": "用一至三句话客观概括会议中实际讨论的主题。",
    },
    {
        "key": "member_progress",
        "title": "各成员工作进展",
        "kind": "list",
        "required": False,
        "guidance": "按成员整理已明确汇报的工作进展和当前状态。",
    },
    {
        "key": "experimental_results",
        "title": "实验结果与数据结论",
        "kind": "list",
        "required": False,
        "guidance": "只记录逐字稿中明确出现的实验结果、数据和结论。",
    },
    {
        "key": "suggestions",
        "title": "建议",
        "kind": "list",
        "required": False,
        "guidance": "整理导师或参会者明确提出的建议，不改写成决定。",
    },
    {
        "key": "decisions",
        "title": "已形成的决定",
        "kind": "list",
        "required": False,
        "guidance": "只记录会议中明确确认或达成一致的决定。",
    },
    {
        "key": "action_items",
        "title": "待办事项",
        "kind": "actions",
        "required": False,
        "guidance": (
            "只收录明确分派、承诺或要求执行的任务，保留负责人、"
            "任务、截止日期和证据时间。"
        ),
    },
    {
        "key": "open_questions",
        "title": "未解决问题与风险",
        "kind": "list",
        "required": False,
        "guidance": "整理仍未解决的问题、分歧和已经明确提到的风险。",
    },
    {
        "key": "next_followups",
        "title": "下次会议跟进",
        "kind": "list",
        "required": False,
        "guidance": "记录下次会议需要继续确认或检查的事项。",
    },
)
CORE_SECTION_KEYS = {
    str(section["key"]) for section in CORE_SECTION_DEFINITIONS
}


class MinutesTemplateError(ValueError):
    pass


def default_minutes_template() -> dict[str, Any]:
    now = utc_now()
    return {
        "id": DEFAULT_TEMPLATE_ID,
        "name": "实验室组会",
        "description": (
            "适合课题组周会，覆盖成员进展、实验结果、建议、决定、"
            "待办和后续跟进。"
        ),
        "instructions": (
            "优先保留实验数据、结论边界、导师建议以及需要继续验证的事项。"
        ),
        "sections": [
            {
                "key": section["key"],
                "title": section["title"],
                "kind": section["kind"],
                "guidance": section["guidance"],
            }
            for section in CORE_SECTION_DEFINITIONS
        ],
        "is_builtin": True,
        "created_at": now,
        "updated_at": now,
    }


def normalize_minutes_template(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    template_id = str(value.get("id") or "").strip()
    if not _SAFE_TEMPLATE_ID.fullmatch(template_id):
        raise MinutesTemplateError("模板 ID 无效")
    name = str(value.get("name") or "").strip()
    if not name:
        raise MinutesTemplateError("模板名称不能为空")
    if len(name) > 80:
        raise MinutesTemplateError("模板名称不能超过 80 个字符")
    description = str(value.get("description") or "").strip()
    if len(description) > 500:
        raise MinutesTemplateError("模板说明不能超过 500 个字符")
    instructions = str(value.get("instructions") or "").strip()
    if len(instructions) > 8_000:
        raise MinutesTemplateError("模板要求不能超过 8000 个字符")

    raw_sections = value.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise MinutesTemplateError("模板至少需要会议摘要章节")
    if len(raw_sections) > len(CORE_SECTION_DEFINITIONS) + MAX_CUSTOM_SECTIONS:
        raise MinutesTemplateError("模板章节数量过多")

    sections: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    summary_count = 0
    actions_count = 0
    for raw in raw_sections:
        if not isinstance(raw, Mapping):
            raise MinutesTemplateError("模板章节格式无效")
        key = str(raw.get("key") or "").strip()
        title = str(raw.get("title") or "").strip()
        kind = str(raw.get("kind") or "").strip()
        guidance = str(raw.get("guidance") or "").strip()
        if not _SAFE_SECTION_KEY.fullmatch(key):
            raise MinutesTemplateError(f"章节键无效：{key or '未填写'}")
        if key in seen_keys:
            raise MinutesTemplateError(f"章节重复：{key}")
        if not title:
            raise MinutesTemplateError("章节名称不能为空")
        if len(title) > 80:
            raise MinutesTemplateError("章节名称不能超过 80 个字符")
        if kind not in _VALID_SECTION_KINDS:
            raise MinutesTemplateError(f"章节类型无效：{kind}")
        if len(guidance) > 1_000:
            raise MinutesTemplateError("单个章节要求不能超过 1000 个字符")
        if kind == "summary":
            summary_count += 1
            if key != "summary":
                raise MinutesTemplateError("摘要章节必须使用 summary 字段")
        if kind == "actions":
            actions_count += 1
            if key != "action_items":
                raise MinutesTemplateError(
                    "待办章节必须使用 action_items 字段"
                )
        seen_keys.add(key)
        sections.append(
            {
                "key": key,
                "title": title,
                "kind": kind,
                "guidance": guidance,
            }
        )
    if summary_count != 1:
        raise MinutesTemplateError("模板必须且只能包含一个会议摘要章节")
    if actions_count > 1:
        raise MinutesTemplateError("模板只能包含一个待办事项章节")
    if sections[0]["key"] != "summary":
        raise MinutesTemplateError("会议摘要必须是模板的第一个章节")

    created_at = str(value.get("created_at") or utc_now())
    updated_at = str(value.get("updated_at") or utc_now())
    return {
        "id": template_id,
        "name": name,
        "description": description,
        "instructions": instructions,
        "sections": sections,
        "is_builtin": bool(value.get("is_builtin", False)),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def build_minutes_template_from_form(
    form: Mapping[str, Any],
    *,
    template_id: str | None = None,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    sections: list[dict[str, str]] = []
    for definition in CORE_SECTION_DEFINITIONS:
        key = str(definition["key"])
        included = bool(definition["required"]) or (
            str(form.get(f"include_{key}") or "").lower()
            in {"on", "true", "1", "yes"}
        )
        if not included:
            continue
        title = str(
            form.get(f"title_{key}") or definition["title"]
        ).strip()
        sections.append(
            {
                "key": key,
                "title": title,
                "kind": str(definition["kind"]),
                "guidance": str(definition["guidance"]),
            }
        )

    custom_lines = str(form.get("custom_sections") or "").splitlines()
    custom_index = 0
    for raw_line in custom_lines:
        line = raw_line.strip()
        if not line:
            continue
        custom_index += 1
        if custom_index > MAX_CUSTOM_SECTIONS:
            raise MinutesTemplateError(
                f"自定义章节最多 {MAX_CUSTOM_SECTIONS} 个"
            )
        title, separator, guidance = line.partition("|")
        clean_title = title.strip()
        clean_guidance = guidance.strip() if separator else ""
        if not clean_title:
            raise MinutesTemplateError("自定义章节名称不能为空")
        sections.append(
            {
                "key": f"custom_{custom_index:02d}",
                "title": clean_title,
                "kind": "list",
                "guidance": (
                    clean_guidance
                    or f"整理与“{clean_title}”直接相关的明确内容。"
                ),
            }
        )

    record = {
        "id": template_id or uuid4().hex,
        "name": str(form.get("name") or ""),
        "description": str(form.get("description") or ""),
        "instructions": str(form.get("instructions") or ""),
        "sections": sections,
        "is_builtin": False,
        "created_at": (
            str(existing.get("created_at"))
            if existing and existing.get("created_at")
            else now
        ),
        "updated_at": now,
    }
    return normalize_minutes_template(record)


def minutes_template_snapshot(
    template: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = normalize_minutes_template(template)
    return {
        "id": normalized["id"],
        "name": normalized["name"],
        "description": normalized["description"],
        "instructions": normalized["instructions"],
        "sections": copy.deepcopy(normalized["sections"]),
    }


def minutes_template_form_values(
    template: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = normalize_minutes_template(template)
    section_by_key = {
        section["key"]: section for section in normalized["sections"]
    }
    custom_lines = []
    for section in normalized["sections"]:
        if section["key"] in CORE_SECTION_KEYS:
            continue
        custom_lines.append(
            f"{section['title']} | {section['guidance']}".rstrip()
        )
    return {
        "id": normalized["id"],
        "name": normalized["name"],
        "description": normalized["description"],
        "instructions": normalized["instructions"],
        "included": set(section_by_key),
        "titles": {
            definition["key"]: (
                section_by_key.get(str(definition["key"]), {}).get(
                    "title", definition["title"]
                )
            )
            for definition in CORE_SECTION_DEFINITIONS
        },
        "custom_sections": "\n".join(custom_lines),
        "is_builtin": normalized["is_builtin"],
    }


def blank_minutes_template_form() -> dict[str, Any]:
    default = default_minutes_template()
    values = minutes_template_form_values(default)
    values.update(
        {
            "id": "",
            "name": "",
            "description": "",
            "instructions": "",
            "is_builtin": False,
        }
    )
    return values
