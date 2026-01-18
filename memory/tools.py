from datetime import datetime
from langchain_core.tools import tool

@tool
def get_current_time():
    """Get the current real-time date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# tools_schema = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_time",
#             "description": "Get the current real-time date and time",
#             "parameters": {
#                 "type": "object",
#                 "properties": {}, 
#                 "required": [],
#             },
#         },
#     }
# ]
# not needed with langraph,it will auto generate.

tools_list = [get_current_time]