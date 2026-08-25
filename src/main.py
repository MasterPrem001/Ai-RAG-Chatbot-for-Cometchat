"""
CLI Interface for Aster & Row RAG Support Agent.

A beautiful Claude Code-like REPL using native ANSI formatting.
"""

# Prevent TensorFlow import issues
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys

from src.agent import Agent
from src.session import Session
from src.logger import StructuredLogger
from src import config

# ANSI Formatting Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def print_banner():
    # Clear screen for a fresh start (cross-platform)
    os.system('cls' if os.name == 'nt' else 'clear')
    print()
    print(f"{BLUE}╭────────────────────────────────────────────────────────────╮{RESET}")
    print(f"{BLUE}│{RESET}  {BOLD}Aster & Row Customer Support Agent{RESET}                        {BLUE}│{RESET}")
    print(f"{BLUE}├────────────────────────────────────────────────────────────┤{RESET}")
    
    primary = f"Gemini ({config.GEMINI_MODEL})" if getattr(config, 'GEMINI_API_KEY', None) else "None"
    fallback = f"Gemini ({config.GEMINI_BACKUP_MODEL})" if getattr(config, 'GEMINI_BACKUP_MODEL', None) else "None"
    
    print(f"{BLUE}│{RESET}  {DIM}Primary LLM:{RESET} {primary:<44} {BLUE}│{RESET}")
    print(f"{BLUE}│{RESET}  {DIM}Fallback LLM:{RESET} {fallback:<43} {BLUE}│{RESET}")
    print(f"{BLUE}├────────────────────────────────────────────────────────────┤{RESET}")
    print(f"{BLUE}│{RESET}  {BOLD}Commands:{RESET}                                                 {BLUE}│{RESET}")
    print(f"{BLUE}│{RESET}    {CYAN}quit / exit{RESET}  - End the session                          {BLUE}│{RESET}")
    print(f"{BLUE}│{RESET}    {CYAN}reset{RESET}        - Start a new conversation                 {BLUE}│{RESET}")
    print(f"{BLUE}│{RESET}    {CYAN}debug{RESET}        - Toggle debug trace mode                  {BLUE}│{RESET}")
    print(f"{BLUE}╰────────────────────────────────────────────────────────────╯{RESET}")
    print()


def format_response(response: str, sources: list[str], handoff: bool):
    """Format the agent's response for display."""
    print()
    print(f"{BOLD}{response}{RESET}")

    if sources:
        print()
        print(f"  {DIM}{BOLD}Sources:{RESET}")
        for src in sources:
            print(f"  {DIM}• {src}{RESET}")

    if handoff:
        print()
        print(f"{YELLOW}╭─ ⚠️  Escalation Triggered ─────────────────────────────────╮{RESET}")
        print(f"{YELLOW}│{RESET}  I'd recommend reaching out to our support team for       {YELLOW}│{RESET}")
        print(f"{YELLOW}│{RESET}  further assistance with this matter.                     {YELLOW}│{RESET}")
        print(f"{YELLOW}╰────────────────────────────────────────────────────────────╯{RESET}")

    print()
    print(f"{DIM}──────────────────────────────────────────────────────────────{RESET}")


def main():
    print_banner()

    # Initialize agent
    sys.stdout.write(f"{DIM}Initializing agent...{RESET}\r")
    sys.stdout.flush()
    try:
        agent = Agent()
        # Clear the initializing line
        sys.stdout.write("                     \r") 
        sys.stdout.flush()
    except Exception as e:
        print(f"\n{BOLD}\033[91mError initializing agent:{RESET} {e}")
        print("Make sure your .env file has a valid GEMINI_API_KEY.")
        sys.exit(1)

    # Create session
    session = Session()
    logger = StructuredLogger(session.session_id)
    
    print(f"{DIM}Session:{RESET} {session.session_id}")
    print(f"{DIM}Debug:{RESET} {GREEN}ON{RESET}" if config.DEBUG else f"{DIM}Debug:{RESET} \033[91mOFF{RESET}")
    print(f"{DIM}Log:{RESET} {logger.log_file}")
    print()
    print(f"{DIM}──────────────────────────────────────────────────────────────{RESET}")
    print()

    while True:
        try:
            sys.stdout.write(f"{BOLD}{GREEN}❯{RESET} ")
            sys.stdout.flush()
            user_input = sys.stdin.readline()
            if not user_input:  # EOF
                print(f"\n\n{BOLD}{BLUE}Goodbye!{RESET}")
                break
            user_input = user_input.strip()
        except KeyboardInterrupt:
            print(f"\n\n{BOLD}{BLUE}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\n{BOLD}{BLUE}Goodbye!{RESET}")
            break

        if user_input.lower() == "reset":
            session = Session()
            logger = StructuredLogger(session.session_id)
            print(f"\n{DIM}New session:{RESET} {session.session_id}")
            print(f"{DIM}Log:{RESET} {logger.log_file}\n")
            print(f"{DIM}──────────────────────────────────────────────────────────────{RESET}")
            print()
            continue

        if user_input.lower() == "debug":
            config.DEBUG = not config.DEBUG
            print(f"\n{DIM}Debug mode:{RESET} {GREEN}ON{RESET}" if config.DEBUG else f"\n{DIM}Debug mode:{RESET} \033[91mOFF{RESET}\n")
            print(f"{DIM}──────────────────────────────────────────────────────────────{RESET}")
            print()
            continue

        # Process message
        try:
            sys.stdout.write(f"{CYAN}Thinking...{RESET}\r")
            sys.stdout.flush()
            response, sources, handoff = agent.process_message(
                user_message=user_input,
                session=session,
                logger=logger,
            )
            # Clear thinking text
            sys.stdout.write("           \r")
            sys.stdout.flush()
            
            format_response(response, sources, handoff)

        except Exception as e:
            print(f"\n{BOLD}\033[91mError:{RESET} {e}")
            if config.DEBUG:
                import traceback
                traceback.print_exc()
            print()
            print(f"{DIM}──────────────────────────────────────────────────────────────{RESET}")
            print()


if __name__ == "__main__":
    main()
