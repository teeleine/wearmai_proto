from pydantic import BaseModel

class QueryEvaluatorOutput(BaseModel):
    is_needed: bool
    query: str
    rationale: str

class RunSummaryOutput(BaseModel):
    text_summary: str

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
            "description": "Indicates if a run summary needs to be generated (for long-term analysis or direct summary requests)."
          },
          "GetRawRunData_needed": {
            "type": "boolean",
            "description": "Indicates if raw run data is required (for detailed analysis of specific runs or short periods)."
          },
          "QueryKnowledgeBase_needed": {
            "type": "boolean",
            "description": "Indicates if a query to the knowledge base is needed for general information."
          },
          "GetGroundingAndFactCheckingData_needed": {
            "type": "boolean",
            "description": "Indicates if grounding and fact-checking data from scientific literature is required."
          },
          "PlotVisualisation_needed": {
            "type": "boolean",
            "description": "Indicates if the user has requested a data visualisation (e.g., plot, chart, graph)."
          },
          "query": {
            "type": "string",
            "description": "The query string for the knowledge base if QueryKnowledgeBase_needed is true, otherwise an empty string."
          },
          "fact_checking_query": {
            "type": "string",
            "description": "The query string (as a question) for scientific literature if GetGroundingAndFactCheckingData_needed is true, otherwise an empty string."
          },
          "visualisation_request_message": {
            "type": "string",
            "description": "The user's specific one-sentence request for the visualisation if PlotVisualisation_needed is true, otherwise an empty string."
          },
          "run_ids": {
            "type": "array",
            "description": "A list of run IDs relevant to the request (for GenerateRunSummary_needed, GetRawRunData_needed, or PlotVisualisation_needed), empty if no specific runs are targeted.",
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
          "PlotVisualisation_needed",
          "query",
          "fact_checking_query",
          "visualisation_request_message",
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
    