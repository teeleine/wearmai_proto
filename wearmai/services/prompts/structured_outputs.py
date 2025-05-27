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
      "name": "function_determinant_output",
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

plotly_visualisation_output_format = {
  "type": "json_schema",
  "name": "plotly_express_code_snippet_output",
  "strict": True,
  "schema": {
    "type": "object",
    "properties": {
      "code_snippet": {
        "type": "string",
        "description": "A Python code snippet string suitable for direct use with Plotly Express. This snippet MUST start directly with a Plotly Express function call (e.g., px.scatter(...)). It includes data arguments (like x, y, color, values, names, z) populated with actual Python lists of data (numbers, strings, or list of lists for imshow z) extracted from the input JSON. The snippet also includes a 'title' argument with a descriptive plot title, and a 'labels' argument mapping Plotly Express argument names (e.g., 'x', 'y', 'color') to human-readable names for axes, legends, or color bars."
      }
    },
    "required": [
      "code_snippet"
    ],
    "additionalProperties": False
  }
}
    