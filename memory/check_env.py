import os
import sys

def check_environment():
    """Check if the required environment variables are set."""
    required_vars = ["GOOGLE_API_KEY"]
    missing = []

    for var in required_vars:
        if var not in os.environ or not os.environ[var]:
            missing.append(var)

    if missing:
        print(f"Error: The following required environment variables are not set: {', '.join(missing)}")
        print("\nPlease set these variables. For example:")
        print("export GOOGLE_API_KEY=your-api-key")
        return False
    
    print("All required environment variables are set.")
    return True

if __name__ == "__main__":
    print("Checking environment variables for the Lineage Agent...")
    if not check_environment():
        sys.exit(1)
    
    print("\nTesting import of required modules...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("Successfully imported ChatGoogleGenerativeAI")
    except ImportError as e:
        print(f"Error importing langchain_google_genai: {e}")
        print("Try reinstalling with: pip install langchain-google-genai")
        sys.exit(1)
    
    print("\nTrying to initialize the model...")
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        print("Successfully initialized the model!")
    except Exception as e:
        print(f"Error initializing the model: {e}")
        print("This might indicate an issue with your API key or network connection.")
        sys.exit(1)
        
    print("\nAll checks passed! The environment appears to be correctly set up.")
    sys.exit(0)
