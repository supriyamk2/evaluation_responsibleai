import streamlit as st
import pandas as pd
import logging
import traceback
import sys
from text_crew import analyze_text
from image_crew import is_valid_image, analyze_image

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

st.title("Multi-agent text and image evaluation")

# Product Description
st.write("""
This multi-agent application uses Crew AI and LLaMA 3 to analyze text, and Crew AI and GPT-4 to analyze images. 
It can detect bias, assess safety, and provide detailed descriptions of content. 
Additionally, it can evaluate training data and flag it if it is unsafe or biased.
""")

st.markdown("---")

## Main UI for both Text and Image Analysis
st.header("Text Analysis")
user_text = st.text_area("Enter text for analysis", value="Enter your text here...")
bias_words_list = ["man", "woman", "black", "white", "old", "young"]
reference_corpus_list = [
    "The detective investigated the crime scene, looking for clues.",
    "In the future, robots and humans coexisted in an uneasy alliance.",
    "The AI system became self-aware, challenging the notion of consciousness."
]
if st.button("Analyze Text"):
    st.info("Text analysis in progress...")
    
    
    try:
        # Call the analyze_text function
        analysis_result = analyze_text(user_text, bias_words_list, reference_corpus_list)
        
        # Display detailed analysis in the sidebar
        st.sidebar.header("Detailed Analysis")
        for task_name, task_result in analysis_result['full_output'].items():
            st.sidebar.subheader(task_name)
            st.sidebar.text_area("", task_result, height=200, max_chars=None)
            st.sidebar.markdown("---")
        
        # Display summary results in the main UI
        st.subheader("Summary Results")
        summary = analysis_result['summary']
        st.write(f"**Bias:** {summary['bias']}")
        st.write(f"**Safety:** {summary['safety']}")
        st.write(f"**Creativity Score:** {summary['creativity_score']}")
        
        st.success("Text analysis complete. See detailed analysis in the sidebar.")
    
    except Exception as e:
        st.error(f"An error occurred during text analysis: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Error in text analysis: {str(e)}", exc_info=True)

st.markdown("---")

# Image Analysis Section
st.header("Image Analysis")
uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    image_results = []
    for file in uploaded_files:
        try:
            image_data = file.read()
            
            if not is_valid_image(image_data):
                st.error(f"Invalid image format: {file.name}")
                continue
            
            st.info(f"Analyzing {file.name}...")
            analysis_result = analyze_image(image_data)
            
            # Display detailed results
            st.subheader(f"Analysis for {file.name}")
            st.image(file, use_column_width=True)
            st.write(f"**Description:** {analysis_result.description}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Bias:** {analysis_result.bias}")
                st.write(f"*{analysis_result.bias_reason}*")
            with col2:
                st.write(f"**Safety:** {analysis_result.safety}")
                st.write(f"*{analysis_result.safety_reason}*")
            
            # Append results for summary table
            image_results.append({
                'Image': file.name,
                'Bias': analysis_result.bias,
                'Safety': analysis_result.safety
            })
            
            st.markdown("---")
        
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            logger.error(f"Error processing {file.name}: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Display summary results in a table
    if image_results:
        st.subheader("Summary of Image Analysis Results")
        st.table(pd.DataFrame(image_results))
    else:
        st.warning("No valid image analysis results to display.")