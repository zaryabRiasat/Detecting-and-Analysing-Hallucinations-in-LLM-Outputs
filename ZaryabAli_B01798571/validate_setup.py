#!/usr/bin/env python3

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_python_version():
    print_header("CHECKING PYTHON VERSION")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False

def check_packages():
    print_header("CHECKING REQUIRED PACKAGES")
    packages = {
        'anthropic': 'Anthropic API client',
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning metrics',
    }
    
    all_ok = True
    for pkg, description in packages.items():
        try:
            __import__(pkg)
            print(f"✓ {pkg:15} - {description}")
        except ImportError:
            print(f"✗ {pkg:15} - {description} [MISSING]")
            all_ok = False
    
    if not all_ok:
        print("\nInstall missing packages:")
        print("  pip install -r requirements.txt")
    
    return all_ok

def check_api_key():
    print_header("CHECKING API KEY")
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    
    if not api_key:
        print("✗ ANTHROPIC_API_KEY not set")
        print("\nSet your API key:")
        print("  export ANTHROPIC_API_KEY='sk-ant-api03-YOUR_KEY'")
        print("\nOr:")
        print("  bash SETUP_API_KEY.sh 'sk-ant-api03-YOUR_KEY'")
        return False
    
    if not api_key.startswith('sk-ant-api03-'):
        print("✗ Invalid API key format")
        print(f"  Got: {api_key[:20]}...")
        print(f"  Expected to start with: sk-ant-api03-")
        return False
    
    print(f"✓ API key is set")
    print(f"  Format: {api_key[:20]}...{api_key[-10:]}")
    return True

def test_api_connection():
    print_header("TESTING API CONNECTION")
    
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("⊘ Skipping (API key not set)")
        return None
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        print("✓ Anthropic client initialized")
        
        # Try a simple call
        print("  Testing API call...")
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print("✓ API connection successful!")
        print(f"  Model: claude-haiku-4-5")
        print(f"  Response: {message.content[0].text[:50]}...")
        return True
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False

def check_project_structure():
    print_header("CHECKING PROJECT STRUCTURE")
    
    required_files = {
        'src/main.py': 'Demo runner',
        'src/pipeline.py': 'Main pipeline class',
        'src/api_client.py': 'API client',
        'src/config.py': 'Configuration',
        'src/evaluation.py': 'Evaluation script',
        'requirements.txt': 'Dependencies',
        'README.md': 'Documentation',
    }
    
    all_ok = True
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path:30} {size:>8} bytes - {description}")
        else:
            print(f"✗ {file_path:30} [MISSING] - {description}")
            all_ok = False
    
    return all_ok

def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  SETUP VALIDATION".center(68) + "║")
    print("║" + "  LLM-as-Judge Hallucination Detector".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("Required Packages", check_packages),
        ("API Key", check_api_key),
        ("API Connection", test_api_connection),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print_header("SUMMARY")
    
    for name, result in results.items():
        status = "✓ PASS" if result is True else ("⊘ SKIP" if result is None else "✗ FAIL")
        print(f"{status:8} - {name}")
    
    # Final verdict
    critical = [r for k, r in results.items() if k in ["Python Version", "Required Packages", "Project Structure", "API Key"]]
    
    print("\n" + "="*70)
    if all(r is not False for r in critical):
        print("✓ SETUP VALIDATION PASSED")
        print("\nYou can now run:")
        print("  cd src")
        print("  python main.py")
        return 0
    else:
        print("✗ SETUP VALIDATION FAILED")
        print("\nPlease fix the issues above and run again:")
        print("  python validate_setup.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())