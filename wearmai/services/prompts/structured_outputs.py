from pydantic import BaseModel

class QueryEvaluatorOutput(BaseModel):
    is_needed: bool
    query: str
    rationale: str

class ConversationSummaryOutput(BaseModel):
    conversation_summary: str

class FunctionDeterminantOutput(BaseModel):
    GenerateRunSummary_needed: bool
    GetRawRunData_needed: bool
    QueryKnowledgeBase_needed: bool
    GetGroundingAndFactCheckingData_needed: bool
    fact_checking_query: str
    query: str
    run_ids: list[int]


function_determinant_json_format = {
      "type": "json_schema",
      "name": "run_summary_request",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "GenerateRunSummary_needed": {
            "type": "boolean",
            "description": "Indicates if a run summary needs to be generated."
          },
          "GetRawRunData_needed": {
            "type": "boolean",
            "description": "Indicates if raw run data is required."
          },
          "QueryKnowledgeBase_needed": {
            "type": "boolean",
            "description": "Indicates if a query to the knowledge base is needed."
          },
          "GetGroundingAndFactCheckingData_needed": {
            "type": "boolean",
            "description": "Indicates if grounding and fact-checking data is required."
          },
          "query": {
            "type": "string",
            "description": "The main query string to execute."
          },
          "fact_checking_query": {
            "type": "string",
            "description": "The query string specifically for fact-checking purposes."
          },
          "run_ids": {
            "type": "array",
            "description": "A list of run IDs associated with the request.",
            "items": {
              "type": "number"
            }
          }
        },
        "required": [
          "GenerateRunSummary_needed",
          "GetRawRunData_needed",
          "QueryKnowledgeBase_needed",
          "GetGroundingAndFactCheckingData_needed",
          "query",
          "fact_checking_query",
          "run_ids"
        ],
        "additionalProperties": False
      }
    }
    