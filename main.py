# Python Code Assistant - Design Documentation

## Core Architecture
# The Python Code Assistant is built as a Streamlit web application that integrates Google's Gemini Pro model through the LangChain framework.
# The application follows a simple flow: users input their API key and code requirements, the system validates the key, 
# processes the request through the LLM, and displays generated code with optional test results. The core strength lies in its modular design, 
# separating concerns between UI handling, API authentication, code generation, and response processing.

## Design Choices and Implementation
# The implementation prioritizes reliability and user experience through several key features. 
# A retry mechanism handles temporary API failures, while comprehensive error handling ensures graceful degradation when issues occur. 
# The prompt template is structured to generate consistent, well-documented Python code following PEP 8 guidelines and includes proper error handling. 
# The response format is strictly defined using tags ([CODE] and [TEST RESULTS]) to ensure reliable parsing and display of results.

## Current Limitations and Assumptions
# The system operates under several practical assumptions: users have valid API keys and basic Python knowledge, 
# and moderate query complexity is expected. The current implementation handles single user sessions and 
# processes one request at a time. Memory management is basic, with simple cleanup after each response processing. 
# The application assumes reasonable response times from the API and doesn't currently implement caching or advanced optimization.

## Future Development Path
# The most impactful improvements would focus on three areas: enhanced functionality, reliability, and user experience. 
# Key additions could include support for multiple programming languages, interactive code editing, and result caching for similar queries. 
# The security layer could be strengthened with proper API key encryption and rate limiting. User experience could be improved with loading indicators, 
# syntax highlighting for test results, and one-click code copying functionality. 
# These improvements would maintain the application's simplicity while expanding its capabilities and reliability.

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import re
import time

# Validate the API key
def valid_api(apikey):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=apikey)
        ans = llm.invoke("test input")
        return True if ans else False
    except Exception:
        return False

# Retry mechanism for LLM invocation
def invoke_with_retry(chain, session_id, query, testcase, retries=3, delay=2):
    for attempt in range(retries):
        try:
            input_data = {"query": query, "testcases": testcase}
            response = chain.run(input_data)
            return response
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)  # Retry delay
            else:
                st.error(f"Agent failed after {retries} attempts: {e}")
                return None

#main function
def main():
    # Streamlit UI
    st.title("Python Code Assistant")
    api_key = st.text_input("Enter your Gemini API key", type="password") # Your API key

    if api_key:  # Only proceed when API key is entered
        if valid_api(api_key):  # Check if the entered API key is valid
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=api_key)

            # the prompt template
            prompt = PromptTemplate(
                input_variables=["query", "testcases"],
                template=(
                    """ You are a Python programming expert. 
                        Generate clean, efficient, and well-documented Python code based on the user's requirements.

                        Requirements: {query}

                        Please follow these guidelines:
                        1. Write well-documented code with clear docstrings
                        2. Include appropriate error handling
                        3. Use type hints where relevant
                        4. Follow PEP 8 style guidelines
                        5. Handle edge cases

                        {testcases}

                        Format your response exactly as follows:

                        [CODE]
                        <Write your Python code here>
                        [END CODE]

                        [TEST RESULTS]
                        <Show test results if {testcases} provided>
                        <Return None if no {testcases} provided>
                        <If {testcases} provided is invalid return Invalid testcase>
                        [END TEST RESULTS]

                        Important:
                        - If test cases are provided, show for each test:
                        * Input: <actual input>
                        * Expected: <expected output>
                        * Result: <actual output>
                        * Status: PASS/FAIL
                        - If no test cases are provided, only show the code section
                        - Don't explain the code unless specifically asked
                        - Don't show multiple solutions unless requested
                        - Don't add any text outside the specified format
                    """
                ),
            )

            # LLM chain
            chain = LLMChain(llm=llm, prompt=prompt)

            # Function to extract code blocks and test result from AI response
            def extract_code_and_tests(response: str):
                """
                Extract both code and test results from the AI response.
                
                Args:
                    response (str): Raw response from the AI
                    
                Returns:
                    tuple: (code, test_results) where both are strings
                """
                # Extract code between [CODE] tags
                code_match = re.search(r'\[CODE\](.*?)\[END CODE\]', response, re.DOTALL)
                code = code_match.group(1).strip() if code_match else "No code found."
                
                # Extract test results between [TEST RESULTS] tags
                test_match = re.search(r'\[TEST RESULTS\](.*?)\[END TEST RESULTS\]', response, re.DOTALL)
                test_results = test_match.group(1).strip() if test_match else ""
                
                return code, test_results

            # Inputs
            st.header("Enter Your Query")
            user_input = st.text_input("Query", placeholder="e.g., Check if a string is a palindrome")
            user_testcase = st.text_input("Testcases (Optional)", placeholder="Sample valid testcases")

            if user_input:
                try:
                    # Always provide a default value for `testcases`
                    testcase_value = user_testcase if user_testcase else "No testcases provided"
                    
                    # Invoke chain and process response
                    response = invoke_with_retry(
                        chain=chain, 
                        session_id="default_session", 
                        query=user_input, 
                        testcase=testcase_value
                    )
                    if response:
                        code, test_results = extract_code_and_tests(response)
                        
                        # Display the code
                        st.subheader("Generated Code:")
                        st.code(code, language="python", line_numbers=True)
                        
                        # Display test results if they exist
                        if test_results != 'None':
                            st.subheader("Test Results:")
                            st.write(test_results)
                    else:
                        st.error("Failed to process the query.")
                    
                    # release memory
                    del response

                except KeyError as e:
                    st.error(f"Error: Missing key in response - {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
        else:
             st.error("Invalid API Key. Please try again.")
    else:
        st.info("Please enter your Gemini API key to start.")

if __name__ == "__main__":
    main()










