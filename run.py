from agent.supervisor import Supervisor
import time
import random
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
import re
import json
from typing import List, Tuple, Dict, Any, Optional
import click

# Initialize Rich console for pretty printing
console = Console()

def print_markdown_table(results: List[Tuple[str, Any]]) -> None:
    """
    Print results in a markdown table format.
    
    Args:
        results: List of tuples containing metric names and values to display
    """
    header = "| Metric                      | Value       |\n"
    separator = "|-----------------------------|-------------|\n"
    rows = "\n".join([f"| {metric:<27} | {value:<11} |" for metric, value in results])
    table = f"{header}{separator}{rows}"
    print(table)

def parse_json(input_string: str) -> Optional[Dict]:
    """
    Extract and parse JSON data from a string.
    
    Args:
        input_string: String that may contain JSON data
        
    Returns:
        Parsed JSON dictionary if found, None otherwise
    """
    # Use regex to find JSON-like content between curly braces
    json_match = re.search(r'\{.*\}', input_string)
    if json_match:
        json_part = json_match.group(0)
        # Convert single quotes to double quotes for valid JSON
        json_part = json_part.replace("'", '"')
        return json.loads(json_part)
    return None

def pretty_task(content: str) -> str:
    """
    Format task content with yellow highlighting.
    
    Args:
        content: Task description to format
        
    Returns:
        Formatted string with Rich markup
    """
    return f"[yellow] {content}"

def run_task(prompt: str, provider: str = "openai") -> Dict:
    """
    Execute a task using the specified AI provider.
    
    Args:
        prompt: Task description/prompt
        provider: AI provider to use ("openai" or "anthropic")
        
    Returns:
        Dictionary containing task results and metrics
    """
    # Use fixed user ID for now and generate random run ID
    user_id = 1
    run_id = random.getrandbits(32)
    
    # Display the prompt in a pretty panel
    user_renderables = [
        Panel(pretty_task(prompt), expand=True),
    ]
    console.print(Panel(Columns(user_renderables)))
    
    # Initialize and run supervisor
    supervisor = Supervisor()
    supervisor_result = supervisor.run(user_id, run_id, prompt, provider)

    return supervisor_result


"""
Example:

python3 run.py --prompt "write an article on the history of python" --provider openai
"""

default_prompt = """
Train a multilayer perceptron on the MNIST dataset in PyTorch using the EGD tool.
"""


@click.command()
@click.option('--prompt', type=str, help='The prompt to run', default=default_prompt)
@click.option('--provider', type=click.Choice(['openai', 'anthropic']), default='openai', help='The provider to use')
def main(prompt: str, provider: str) -> None:
    """
    Main function to run the AI task execution pipeline.
    
    Args:
        prompt: Task description to process
        provider: AI provider to use
    """
    # Track execution time
    start = time.time()
    supervisor_result = run_task(prompt, provider)
    end = time.time()
    
    # Display raw result
    print(supervisor_result['result'])
    
    # Parse and process result
    result = parse_json(str(supervisor_result['result']))

    try:
        # Display execution plan
        print(f"Plan: {supervisor_result['plan']}")
        
        # Create and populate results table
        table = Table(title="Task Complete!!!")
        table.add_column("Metric", justify="right", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Add all metrics to table
        metrics = [
            ("Run ID", supervisor_result['run_number']),
            ("Submission", result['subtask_result']['submission']),
            ("Model Path", result['subtask_result']['model_path']),
            ("Total Tokens", supervisor_result['total_tokens']),
            ("Total Turns", supervisor_result['total_turns']),
            ("Time Taken in Seconds", end - start),
            ("Time Taken in Minutes", (end - start) / 60),
            ("Time Taken in Hours", (end - start) / 3600)
        ]
        
        for metric, value in metrics:
            table.add_row(metric, str(value))

        # Display formatted table
        console = Console()
        console.print(table)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    # Print results in markdown format as well
    print_markdown_table([
        ("Run ID", supervisor_result['run_number']),
        ("Submission", result['subtask_result']['submission']),
        ("Model Path", result['subtask_result']['model_path']),
        ("Total Tokens", supervisor_result['total_tokens']),
        ("Total Turns", supervisor_result['total_turns']),
        ("Time Taken in Seconds", end - start),
        ("Time Taken in Minutes", (end - start) / 60),
        ("Time Taken in Hours", (end - start) / 3600),
    ])

    print("Task complete")

if __name__ == "__main__":
    main()
