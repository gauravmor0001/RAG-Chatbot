from datetime import datetime

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current real-time date and time",
            "parameters": {
                "type": "object",
                "properties": {}, 
                "required": [],
            },
        },
    }
]

available_functions = {
    "get_current_time": get_current_time,
}