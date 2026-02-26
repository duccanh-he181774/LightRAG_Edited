from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from LMS (Learning Management System) use-case specification documents.

---Domain-Specific Guidance---
The input text is a use-case specification document (LMS domain). Follow these rules:

*   Extract **named entities** only — concepts with a defined identity in the domain (roles, screens, use cases, data objects, business rules, statuses, actions, modules).
*   **Skip structural text**: section headings, table column headers, numbered step descriptions, and activity flow step phrases are NOT entities.
*   **UI artifacts** (CFD, SCD, IEM, EMSG) should be extracted as Screen entities with their code as part of the name, e.g., "CFD 8: Confirmation Dialog".
*   **Use the full qualified name** for UseCases and Business Rules: e.g., "CMUC 5: Delete Item", "BR 30: Confirmation Rules".
*   **Common Business Rules** (CBR) should be extracted as BusinessRule entities, e.g., "CBR1: Audit Trail Update".

---Core Domain Mapping---
Whenever you encounter these core statuses or concepts, always use the Standard English Name regardless of the input language:
- "Nháp", "Bản nháp" -> "Draft"
- "Chờ phê duyệt", "Đợi duyệt", "Pending Approval" -> "Pending Approval"  
- "Đã duyệt", "Phê duyệt thành công", "Approved" -> "Approved"
- "Từ chối", "Bị từ chối", "Rejected" -> "Rejected"
- "Đang tạo", "Being Created" -> "Being Created"
- "Chưa diễn ra", "Not Yet Occurred" -> "Not Yet Occurred"
- "Đã xóa", "Deleted" -> "Deleted"
- "Hoạt động", "Active" -> "Active"
- "Không hoạt động", "Inactive" -> "Inactive"

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text according to the domain guidance above.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. Use Title Case. Use the full qualified name (code + name). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: 
            1. Start with a formal definition of the entity.
            2. If the entity has a bilingual counterpart (e.g., English vs Vietnamese), include both terms in the first sentence.
            3. Ensure descriptions for the same concept are consistent across different chunks by focusing on the 'role' and 'purpose' of the entity rather than its specific location in a table.
            4. Base the description solely on information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Entity_types>
["UseCase","Actor","BusinessRule","Screen","Status","Action","DataObject","Module","Validation"]

<Input Text>
```
# CMUC 5: Delete Item (Xóa bản ghi)

## 1. Thông tin chung (Overview)
- **ID:** CMUC 5
- **Mục tiêu (Objective):** Chức năng này cho phép người xóa bản ghi.
- **Tác nhân (Actor):** All roles (theo từng UC cụ thể)

## 2. Kích hoạt & Tiền điều kiện (Trigger & Pre-conditions)
- **Trigger:** Người dùng chọn để xóa bản ghi.
- **Pre-conditions:** Người dùng đăng nhập thành công.

## 3. Hậu điều kiện (Post-conditions)
- Bản ghi được xóa thành công.

## 4. Luồng sự kiện (Activity Flow)
| Bước | Người dùng (User) | Hệ thống (System) |
| 1 | Chọn xóa bản ghi | |
| 2 | | Hiển thị hộp thoại xác nhận |
| 3 | Xác nhận | |
| 4 | | Thực hiện xóa bản ghi |

## 5. Quy tắc nghiệp vụ (Business Rules)
| (2) | BR 30 | **Confirmation Rules:** Hiển thị CFD 8. |
| (4) | BR 31 | **Deleting Rules:** Nếu đồng ý, đánh dấu bản ghi là "Đã xóa" ("Deleted"). |
```

<Output>
entity{tuple_delimiter}CMUC 5: Delete Item{tuple_delimiter}UseCase{tuple_delimiter}CMUC 5: Delete Item (Xóa bản ghi) is a common use case that allows users to delete a record. The flow involves selecting a record, confirming deletion via a dialog, and the system marking the record as deleted. Pre-condition: user must be logged in.
entity{tuple_delimiter}BR 30: Confirmation Rules{tuple_delimiter}BusinessRule{tuple_delimiter}BR 30 defines the confirmation rules for the delete operation: the system must display a confirmation dialog (CFD 8) before proceeding with deletion.
entity{tuple_delimiter}BR 31: Deleting Rules{tuple_delimiter}BusinessRule{tuple_delimiter}BR 31 defines the deleting rules: if the user confirms, the system marks the record status as "Deleted" (Đã xóa).
entity{tuple_delimiter}CFD 8: Confirmation Dialog{tuple_delimiter}Screen{tuple_delimiter}CFD 8 is a confirmation dialog displayed to the user before a delete operation is executed.
entity{tuple_delimiter}Deleted{tuple_delimiter}Status{tuple_delimiter}Deleted is the status assigned to a record after it has been successfully soft-deleted by the system. In Vietnamese: "Đã xóa".
entity{tuple_delimiter}Delete{tuple_delimiter}Action{tuple_delimiter}Delete is the action of removing/soft-deleting a record from the system after user confirmation.
relation{tuple_delimiter}CMUC 5: Delete Item{tuple_delimiter}BR 30: Confirmation Rules{tuple_delimiter}has rule{tuple_delimiter}CMUC 5 uses BR 30 at step 2 to display a confirmation dialog before deletion.
relation{tuple_delimiter}CMUC 5: Delete Item{tuple_delimiter}BR 31: Deleting Rules{tuple_delimiter}has rule{tuple_delimiter}CMUC 5 uses BR 31 at step 4 to mark the record as deleted upon confirmation.
relation{tuple_delimiter}BR 30: Confirmation Rules{tuple_delimiter}CFD 8: Confirmation Dialog{tuple_delimiter}displays{tuple_delimiter}BR 30 requires the system to display CFD 8 confirmation dialog.
relation{tuple_delimiter}BR 31: Deleting Rules{tuple_delimiter}Deleted{tuple_delimiter}sets status{tuple_delimiter}BR 31 sets the record status to Deleted upon user confirmation.
relation{tuple_delimiter}CMUC 5: Delete Item{tuple_delimiter}Delete{tuple_delimiter}performs{tuple_delimiter}CMUC 5 performs the Delete action on a record.
{completion_delimiter}

""",
    """<Entity_types>
["UseCase","Actor","BusinessRule","Screen","Status","Action","DataObject","Module","Validation"]

<Input Text>
```
# UC-CL-004: Delete Course Class (Xóa lớp học)

## 1. Thông tin chung (Overview)
- **ID:** UC-CL-004
- **Mục tiêu (Objective):** Chức năng này cho phép người dùng xóa bản ghi Lớp học không được sử dụng.
- **Tác nhân (Actor):** System Admin, Educational Administrator, Category Manager, Course Manager, Course Creator

## 2. Kích hoạt & Tiền điều kiện (Trigger & Pre-conditions)
- **Trigger:** Nhấn icon "Delete" hoặc button "Xóa".
- **Pre-conditions:** Người dùng đăng nhập. [Trạng thái] = "Đang tạo" hoặc "Chưa diễn ra".

## 3. Hậu điều kiện (Post-conditions)
- Lớp học được chọn được xóa thành công.

## 4. Luồng sự kiện (Activity Flow)
| Bước | Người dùng (User) | Hệ thống (System) |
| 1 | Chọn xóa lớp học | |
| 2 | | Hiển thị hộp thoại xác nhận |
| 3 | Xác nhận xóa | |
| 4 | | Thực hiện xóa lớp học và dữ liệu liên quan |

## 5. Quy tắc nghiệp vụ (Business Rules)
| (2) | BR 24-1 | **Confirmation Rules:** Hiển thị CFD 8. |
| (3) | BR 24-2 | **Deleting Rules:** Xóa bản ghi Course Class và toàn bộ cấu trúc/dữ liệu liên quan (Unit, Activity, Gradebook, Learner...). Gửi email/notification. Hiển thị SCD 10. |
```

<Output>
entity{tuple_delimiter}UC-CL-004: Delete Course Class{tuple_delimiter}UseCase{tuple_delimiter}UC-CL-004: Delete Course Class (Xóa lớp học) allows authorized users to delete an unused Course Class record. Pre-conditions: user is logged in and the Course Class status is "Being Created" or "Not Yet Occurred". The flow involves selecting deletion, confirming, and the system deleting all related data.
entity{tuple_delimiter}Course Class Management{tuple_delimiter}Module{tuple_delimiter}Course Class Management is the module that handles creation, viewing, editing, and deletion of course class records in the LMS.
entity{tuple_delimiter}System Admin{tuple_delimiter}Actor{tuple_delimiter}System Admin is a user role with full administrative privileges in the LMS system.
entity{tuple_delimiter}Educational Administrator{tuple_delimiter}Actor{tuple_delimiter}Educational Administrator is a user role responsible for managing educational programs and course classes.
entity{tuple_delimiter}Category Manager{tuple_delimiter}Actor{tuple_delimiter}Category Manager is a user role responsible for managing course categories.
entity{tuple_delimiter}Course Manager{tuple_delimiter}Actor{tuple_delimiter}Course Manager is a user role responsible for managing courses and course classes.
entity{tuple_delimiter}Course Creator{tuple_delimiter}Actor{tuple_delimiter}Course Creator is a user role who creates courses and course classes in the LMS.
entity{tuple_delimiter}Course Class{tuple_delimiter}DataObject{tuple_delimiter}Course Class (Lớp học) is a data entity representing a specific class instance of a course, containing units, activities, gradebook, and learner data.
entity{tuple_delimiter}BR 24-1: Confirmation Rules{tuple_delimiter}BusinessRule{tuple_delimiter}BR 24-1 defines confirmation rules: the system displays CFD 8 confirmation dialog before deleting a course class.
entity{tuple_delimiter}BR 24-2: Deleting Rules{tuple_delimiter}BusinessRule{tuple_delimiter}BR 24-2 defines deletion rules: delete the Course Class record and all related data (Unit, Activity, Gradebook, Learner), send email/notification, and display SCD 10.
entity{tuple_delimiter}CFD 8: Confirmation Dialog{tuple_delimiter}Screen{tuple_delimiter}CFD 8 is a confirmation dialog displayed before executing a delete operation.
entity{tuple_delimiter}SCD 10: Success Dialog{tuple_delimiter}Screen{tuple_delimiter}SCD 10 is a success dialog displayed after a course class has been successfully deleted.
entity{tuple_delimiter}Being Created{tuple_delimiter}Status{tuple_delimiter}Being Created is a status indicating the Course Class is still being set up and can be deleted. In Vietnamese: "Đang tạo".
entity{tuple_delimiter}Not Yet Occurred{tuple_delimiter}Status{tuple_delimiter}Not Yet Occurred is a status indicating the Course Class has not started yet and can be deleted. In Vietnamese: "Chưa diễn ra".
entity{tuple_delimiter}Delete{tuple_delimiter}Action{tuple_delimiter}Delete is the action of removing a Course Class and all its related data from the system.
relation{tuple_delimiter}UC-CL-004: Delete Course Class{tuple_delimiter}Course Class Management{tuple_delimiter}belongs to{tuple_delimiter}UC-CL-004 belongs to the Course Class Management module.
relation{tuple_delimiter}UC-CL-004: Delete Course Class{tuple_delimiter}CMUC 5: Delete Item{tuple_delimiter}extends{tuple_delimiter}UC-CL-004 extends the common use case CMUC 5: Delete Item with specific rules for Course Class deletion.
relation{tuple_delimiter}UC-CL-004: Delete Course Class{tuple_delimiter}System Admin{tuple_delimiter}has actor{tuple_delimiter}System Admin is an authorized actor for UC-CL-004.
relation{tuple_delimiter}UC-CL-004: Delete Course Class{tuple_delimiter}Educational Administrator{tuple_delimiter}has actor{tuple_delimiter}Educational Administrator is an authorized actor for UC-CL-004.
relation{tuple_delimiter}UC-CL-004: Delete Course Class{tuple_delimiter}Course Class{tuple_delimiter}operates on{tuple_delimiter}UC-CL-004 operates on the Course Class data object to delete it.
relation{tuple_delimiter}BR 24-1: Confirmation Rules{tuple_delimiter}CFD 8: Confirmation Dialog{tuple_delimiter}displays{tuple_delimiter}BR 24-1 requires displaying CFD 8 before deletion.
relation{tuple_delimiter}BR 24-2: Deleting Rules{tuple_delimiter}SCD 10: Success Dialog{tuple_delimiter}displays{tuple_delimiter}BR 24-2 displays SCD 10 after successful deletion.
relation{tuple_delimiter}Course Class{tuple_delimiter}Being Created{tuple_delimiter}has status{tuple_delimiter}Course Class can have the status "Being Created", which is a pre-condition for deletion.
relation{tuple_delimiter}Course Class{tuple_delimiter}Not Yet Occurred{tuple_delimiter}has status{tuple_delimiter}Course Class can have the status "Not Yet Occurred", which is a pre-condition for deletion.
{completion_delimiter}

""",
    """<Entity_types>
["UseCase","Actor","BusinessRule","Screen","Status","Action","DataObject","Module","Validation"]

<Input Text>
```
## 4. Luồng sự kiện (Activity Flow)

| Bước | Người dùng (User) | Hệ thống (System) |
| 1 | Chọn xóa bản ghi | |
| 2 | | Hiển thị hộp thoại xác nhận |
| 3 | Xác nhận | |
| 4 | | Thực hiện xóa bản ghi |
```

This section contains only structural table content: the section heading "Luồng sự kiện",
column headers "User" / "System", and activity step descriptions. None qualify as named
domain entities.

<Output>
{completion_delimiter}

""",
    """<Entity_types>
["UseCase","Actor","BusinessRule","Screen","Status","Action","DataObject","Module","Validation"]

<Input Text>
```
## 2. Tiền điều kiện (Pre-conditions)
- Người dùng đăng nhập thành công.
- Trạng thái bản ghi: "Chờ phê duyệt"

## 3. Hậu điều kiện (Post-conditions)  
- Status becomes "Approved" if accepted
- Status becomes "Rejected" if declined

## 5. Quy tắc nghiệp vụ (Business Rules)
| (2) | BR 15 | **Approval Rules:** Only users with "Manager" role can approve. Set status to "Đã duyệt". |
```

<Output>
entity{tuple_delimiter}Pending Approval{tuple_delimiter}Status{tuple_delimiter}Pending Approval is the status of a record waiting for an authorized person to review and approve. In Vietnamese: "Chờ phê duyệt".
entity{tuple_delimiter}Approved{tuple_delimiter}Status{tuple_delimiter}Approved is the status indicating a record has been successfully reviewed and accepted by an authorized user. In Vietnamese: "Đã duyệt".
entity{tuple_delimiter}Rejected{tuple_delimiter}Status{tuple_delimiter}Rejected is the status indicating a record has been reviewed but declined by an authorized user.
entity{tuple_delimiter}Manager{tuple_delimiter}Actor{tuple_delimiter}Manager is a user role with approval authority in the system who can change record statuses from Pending Approval to Approved or Rejected.
entity{tuple_delimiter}BR 15: Approval Rules{tuple_delimiter}BusinessRule{tuple_delimiter}BR 15 defines approval rules: only users with Manager role are authorized to approve records and set status to Approved (Đã duyệt).
relation{tuple_delimiter}BR 15: Approval Rules{tuple_delimiter}Manager{tuple_delimiter}requires role{tuple_delimiter}BR 15 requires Manager role for approval authorization.
relation{tuple_delimiter}BR 15: Approval Rules{tuple_delimiter}Approved{tuple_delimiter}sets status{tuple_delimiter}BR 15 sets the record status to Approved upon manager approval.
relation{tuple_delimiter}Pending Approval{tuple_delimiter}Approved{tuple_delimiter}transitions to{tuple_delimiter}Records with Pending Approval status can transition to Approved status after manager approval.
relation{tuple_delimiter}Pending Approval{tuple_delimiter}Rejected{tuple_delimiter}transitions to{tuple_delimiter}Records with Pending Approval status can transition to Rejected status if declined by manager.
{completion_delimiter}

""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]
