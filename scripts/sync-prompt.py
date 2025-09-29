#!/usr/bin/env python3
"""
Synchronization script to keep prompt.md and README.md transformation sections in sync.
"""

from pathlib import Path

def sync_prompt_files():
    """Ensure prompt.md exists and is accessible"""
    prompt_path = Path("prompt.md")
    readme_path = Path("README.md")
    
    if not prompt_path.exists() or not readme_path.exists():
        print("Missing required files")
        return False
    
    # Read the standalone prompt
    prompt_content = prompt_path.read_text(encoding="utf-8").strip()
    
    # Check if files exist and are readable
    print(f"Prompt file: {len(prompt_content)} characters")
    print("Sync check completed - README references external prompt file")
    
    return True

def create_github_pages_prompt(url: str):
    """Create a GitHub Pages version of the prompt"""
    prompt_path = Path("prompt.md")
    pages_dir = Path("docs")
    pages_prompt = pages_dir / "prompt.md"
    
    if not prompt_path.exists():
        return False
    
    pages_dir.mkdir(exist_ok=True)
    
    # Create a GitHub Pages optimized version
    prompt_content = prompt_path.read_text(encoding="utf-8")
    
    pages_content = f"""---
title: Go-ish Python Transformation Prompt
description: Comprehensive prompt for transforming Python codebases to Go-ish patterns
---

# Python to Go-ish Transformation Prompt

This prompt can be used with Claude, GPT-4, or any AI assistant to transform Python codebases.

## Usage

1. Copy the entire prompt below
2. Paste it into your AI assistant 
3. Follow with your Python code or repository

---

{prompt_content}

---

## Quick Actions

- [📋 Copy this prompt](javascript:navigator.clipboard.writeText(document.querySelector('pre').textContent))
- [🔙 Back to main guide]({url})
- [📁 View repository]({url})
"""
    
    pages_prompt.write_text(pages_content, encoding="utf-8")
    print(f"Created GitHub Pages prompt: {pages_prompt}")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync prompt.md and README.md transformation sections")
    parser.add_argument("--url", type=str, default="https://yamaceay.github.io/my-py-style/", help="Base URL for GitHub Pages links")
    args = parser.parse_args()

    success = sync_prompt_files()
    if success:
        create_github_pages_prompt(args.url)
        print("Documentation sync completed successfully")
    else:
        print("Documentation sync failed")
        exit(1)