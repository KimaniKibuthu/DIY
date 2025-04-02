import streamlit as st
import math
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import boto3
from llama_index.llms.bedrock import Bedrock
import re
import logging
import io
from PIL import Image
import os
import numpy as np
import time
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from typing import List, Tuple
import dateutil.parser
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from joblib import Parallel, delayed


logging.basicConfig(filename='responses.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Bedrock setup
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='eu-west-1'
)

bedrock_llm = Bedrock(
    model="mistral.mistral-large-2402-v1:0",
    client=bedrock_client,
    streaming=False,
    model_kwargs={
        "temperature": 0.8
    }
)


from crewai import Agent, Task, Crew, LLM
from langchain_openai import ChatOpenAI
from crewai_tools import ScrapeWebsiteTool
import os 
from dotenv import load_dotenv

llm = LLM(model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")

# Instantiate tools
site = 'https://www.safaricom.co.ke'
web_scrape_tool = ScrapeWebsiteTool(website_url=site)

# Create agents
web_scraper_agent = Agent(
    role='Web Scraper',
    goal='Extract and compile all voice and call-related product information from the website',
    backstory='''You are an expert web scraper specialized in extracting telecommunications product information. 
                 Your task is to visit the website and extract all actual content related to voice calls, 
                 calling plans, and voice services. Do not create templates - only extract real content.''',
    tools=[web_scrape_tool],
    verbose=False,
    llm=llm
)




# Define tasks
web_scraper_task = Task(
    description='''Access the website and extract all actual content related to voice and call products. 
                  Include specific prices, features, and terms. Do not create placeholders - only extract 
                  real information that exists on the site.''',
    expected_output='''A detailed compilation of all voice and call-related products and services, 
                      including actual prices, features, and terms found on the website.''',
    agent=web_scraper_agent,
    output_file='data.txt'
)

# Assemble a crew
saf_crew = Crew(
    agents=[web_scraper_agent],
    tasks=[web_scraper_task],
    verbose=False
)


# Instantiate tools
Asite = 'https://www.airtelkenya.com'
Aweb_scrape_tool = ScrapeWebsiteTool(website_url=Asite)

# Create agents
Aweb_scraper_agent = Agent(
    role='Web Scraper',
    goal='Extract and compile all voice and call-related product information from the website',
    backstory='''You are an expert web scraper specialized in extracting telecommunications product information. 
                 Your task is to visit the website and extract all actual content related to voice calls, 
                 calling plans, and voice services. Do not create templates - only extract real content.''',
    tools=[Aweb_scrape_tool],
    verbose=False,
    llm=llm
)

# Define tasks
Aweb_scraper_task = Task(
    description='''Access the website and extract all actual content related to voice and call products. 
                  Include specific prices, features, and terms. Do not create placeholders - only extract 
                  real information that exists on the site.''',
    expected_output='''A detailed compilation of all voice and call-related products and services, 
                      including actual prices, features, and terms found on the website.''',
    agent=Aweb_scraper_agent,
    output_file='data.txt'
)

# Assemble a crew
airtel_crew = Crew(
    agents=[Aweb_scraper_agent],
    tasks=[Aweb_scraper_task],
    verbose=False
)




from crewai import Agent, Task, Crew, LLM
from langchain_openai import ChatOpenAI
from crewai_tools import ScrapeWebsiteTool
import os 
from dotenv import load_dotenv
from datetime import datetime
import concurrent.futures
from tqdm import tqdm  # For progress bar


def saf_online_today(query,llm=llm):
    try:
        from googlesearch import search
    except ImportError: 
        print("No module named 'google' found")

    def scrape_site(site):
        try:
            web_scrape_tool = ScrapeWebsiteTool(website_url=site)
            web_scraper_agent = Agent(
                role='Web Scraper',
                goal='Extract and compile all Issues affecting Safaricom from the website',
                backstory=f'''You are an expert web scraper specialized in extracting issues that negatively affect Safaricom for the date {datetime.today().strftime('%Y-%m-%d')}. 
                             Your task is to visit the website and extract all actual content related to Safaricom.\
                             Do not create templates - only extract real content.''',
                tools=[web_scrape_tool],
                verbose=False,
                llm=llm
            )

            web_scraper_task = Task(
                description=f'''Access the website and extract all actual content related to issues that negatively affect Safaricom for the date {datetime.today().strftime('%Y-%m-%d')}. 
                              Include all specifics. Do not create placeholders - only extract 
                              real information that exists on the site.''',
                expected_output=f'''A detailed compilation of all issues that affect Safaricom found on the website for the date {datetime.today().strftime('%Y-%m-%d')}.''',
                agent=web_scraper_agent,
                output_file='data.txt'
            )

            crew = Crew(
                agents=[web_scraper_agent],
                tasks=[web_scraper_task],
                verbose=False
            )

            result = crew.kickoff()
            return {'url': site, 'content': result, 'status': 'success'}
        except Exception as e:
            return {'url': site, 'content': str(e), 'status': 'error'}

    # Search for URLs
    query2 = f"{query} {datetime.today().strftime('%Y-%m-%d')}"
    urls = list(search(query2, tld="co.in", num=10, stop=10, pause=2))

    # Define the maximum number of concurrent workers
    max_workers = 10  # Adjust this based on your system's capabilities

    # Process URLs in parallel
    outputs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a progress bar
        futures = [executor.submit(scrape_site, url) for url in urls]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scraping websites"):
            result = future.result()
            if result['status'] == 'success':
                outputs.append(result['content'])
            else:
                print(f"Error scraping {result['url']}: {result['content']}")

    # Analyze results
    if outputs:
        analysis = bedrock_llm.complete(
            f'You are media analyst, you need to analyze these outputs {outputs} \
            and create a proper summary of what is happening for Safaricom online'
        )
        print("\nAnalysis:")
        print(analysis)
    else:
        print("No successful scrapes to analyze")
    return analysis


# Adding the new causal analysis functions
def create_causal_graph(features):
    G = nx.DiGraph()
    G.add_nodes_from(features)
    
    # User-related features
    user_features = [f for f in ['Charged_Voice_Users_Pre_Paid', 'Messaging_Users_Pre_Paid', 'Okoa_Jahazi_Internet_Users_Pre_Paid',
                     'One_Month_Active_Pre_Paid', 'One_Month_Active_Unknown', 'Prior_Activations_Pre_Paid',
                     'SMS_P2P_Customers_Pre_Paid', 'Skiza_Users_Pre_Paid', 'Smartphone_Users_Pre_Paid',
                     'Successful_Rvrs_Callers_Pre_Paid', 'Three_Month_Active_Pre_Paid', 'Three_Month_Active_Unknown']
                     if f in features]
    
    # Network-related features
    network_features = [f for f in ['SUM_TCH_Drops_Daily', 'AVG_TCH_Drops_Daily', 'AVG_CSSR_Daily']
                        if f in features]
    
    # Time-related features
    time_features = [f for f in ['is_Weekday', 'is_holiday', 'dayofweek', 'weekend', 'dayofyear', 'sin_day', 'cos_day',
                     'dayofmonth', 'weekofyear', 'end', 'weekofmonth']
                     if f in features]
    
    # User features affect network usage and revenue
    for feature in user_features:
        for target in ['Off_Net', 'On_Net', 'recharge', 'freq', 'Voice_Revenue']:
            if target in features:
                G.add_edge(feature, target)
    
    # Network features affect user behavior and revenue
    for feature in network_features:
        for target in ['Off_Net', 'On_Net', 'Voice_Revenue']:
            if target in features:
                G.add_edge(feature, target)
    
    # Time features affect user behavior and network performance
    for feature in time_features:
        for target in user_features + network_features + ['Off_Net', 'On_Net', 'recharge', 'freq', 'Voice_Revenue']:
            if target in features:
                G.add_edge(feature, target)
    
    # Usage features affect revenue
    for feature in ['Off_Net', 'On_Net', 'recharge', 'freq']:
        if feature in features and 'Voice_Revenue' in features:
            G.add_edge(feature, 'Voice_Revenue')
    
    # Special offers affect revenue
    for feature in ['NEO_VOICE', 'TUNUKIWA_NEO_VOICE', 'TUNUKIWA_VOICE', 'VOICE_OKOA', 'VOICE_ROAMING']:
        if feature in features and 'Voice_Revenue' in features:
            G.add_edge(feature, 'Voice_Revenue')
    
    # Churn and reconnections affect active users
    if 'Three_Month_Churn_Pre_Paid' in features and 'Three_Month_Active_Pre_Paid' in features:
        G.add_edge('Three_Month_Churn_Pre_Paid', 'Three_Month_Active_Pre_Paid')
    if 'Three_Month_Reconnections_Pre_Paid' in features and 'Three_Month_Active_Pre_Paid' in features:
        G.add_edge('Three_Month_Reconnections_Pre_Paid', 'Three_Month_Active_Pre_Paid')
    
    return G



def double_ml_with_weighting(X, y, causal_graph, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def process_fold(train_index, test_index):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        selected_features = X_train_scaled
        X_train_scaled = X_train_scaled[selected_features]
        X_test_scaled = X_test_scaled[selected_features]
        
        model_y = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_y.fit(X_train_scaled, y_train)
        y_residuals = y_test - model_y.predict(X_test_scaled)
        
        feature_importance = {}
        for feature in X_train_scaled.columns:
            parents = list(set(causal_graph.predecessors(feature)) & set(X_train_scaled.columns))
            
            if not parents:
                X_parents = X_train_scaled[[feature]]
                t_residuals = X_test_scaled[feature] - X_train_scaled[feature].mean()
            else:
                X_parents = X_train_scaled[parents]
                model_t = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model_t.fit(X_parents, X_train_scaled[feature])
                t_residuals = X_test_scaled[feature] - model_t.predict(X_test_scaled[parents])
            
            epsilon = 1e-8
            t_residuals_adj = np.where(np.abs(t_residuals) < epsilon, epsilon, t_residuals)
            
            X_weighted = pd.concat([X_test_scaled.drop(columns=[feature])] * 2, ignore_index=True)
            y_weighted = np.concatenate([y_residuals / t_residuals_adj, np.zeros_like(y_residuals)])
            weights = np.concatenate([np.abs(t_residuals_adj), 1 - np.abs(t_residuals_adj)])
            
            mask = np.isfinite(y_weighted) & np.isfinite(weights)
            X_weighted = X_weighted[mask]
            y_weighted = y_weighted[mask]
            weights = weights[mask]
            
            if len(X_weighted) > 0:
                model_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model_final.fit(X_weighted, y_weighted, sample_weight=weights)
                feature_importance[feature] = np.mean(model_final.feature_importances_)
            else:
                feature_importance[feature] = 0
        
        return feature_importance
    
    feature_importances = Parallel(n_jobs=-1)(
        delayed(process_fold)(train_index, test_index)
        for train_index, test_index in kf.split(X)
    )
    
    avg_importances = {}
    for feature in X.columns:
        avg_importances[feature] = np.mean([fold.get(feature, 0) for fold in feature_importances])
    
    return avg_importances

def sensitivity_analysis(X, y, causal_graph, n_iterations=100):
    def process_iteration(X, y, feature):
        X_noisy = X.copy()
        X_noisy[feature] += np.random.normal(0, X[feature].std() * 0.1, size=len(X))
        
        scaler = StandardScaler()
        X_noisy_scaled = pd.DataFrame(scaler.fit_transform(X_noisy), columns=X_noisy.columns)
        
        importance = double_ml_with_weighting(X_noisy_scaled, y, causal_graph, n_splits=2)
        return importance[feature]
    
    sensitivities = {}
    for feature in X.columns:
        importances = Parallel(n_jobs=-1)(
            delayed(process_iteration)(X, y, feature)
            for _ in range(n_iterations)
        )
        sensitivities[feature] = np.std(importances)
    
    return sensitivities

# Regression Discontinuity Design functions
def prepare_data_for_rd(df: pd.DataFrame, cutoff_date: str, outcome: str = 'Voice_Revenue') -> pd.DataFrame:
    df = df.copy()
    df = df.reset_index(drop=True)
    cutoff_date = pd.to_datetime(cutoff_date)
    df['days_from_cutoff'] = (df['date'] - cutoff_date).dt.days
    df['post_treatment'] = (df['days_from_cutoff'] >= 0).astype(int)
    return df


def run_rd_analysis(df: pd.DataFrame, bandwidth: int = 30) -> Tuple[float, float, float]:
    """
    Run regression discontinuity analysis.
    
    :param df: Prepared DataFrame
    :param bandwidth: Number of days before and after the cutoff to include
    :return: Tuple of (effect, standard_error, p_value)
    """
    df_subset = df[(df['days_from_cutoff'] >= -bandwidth) & (df['days_from_cutoff'] <= bandwidth)]
    
    model = smf.ols('Voice_Revenue ~ days_from_cutoff + post_treatment + days_from_cutoff:post_treatment', data=df_subset)
    results = model.fit()
    
    effect = results.params['post_treatment']
    se = results.bse['post_treatment']
    p_value = results.pvalues['post_treatment']
    
    return effect, se, p_value


@st.cache_data
def plot_rd(df: pd.DataFrame, cutoff_date: str, bandwidth: int = 30):
    """
    Plot the regression discontinuity and return as BytesIO object.
    
    :param df: Prepared DataFrame
    :param cutoff_date: The date of the intervention/cutoff
    :param bandwidth: Number of days before and after the cutoff to include
    :return: BytesIO object containing the plot
    """
    df_subset = df[(df['days_from_cutoff'] >= -bandwidth) & (df['days_from_cutoff'] <= bandwidth)]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df_subset['days_from_cutoff'], df_subset['Voice_Revenue'], alpha=0.5)
    
    # Fit lines before and after cutoff
    before = df_subset[df_subset['days_from_cutoff'] < 0]
    after = df_subset[df_subset['days_from_cutoff'] >= 0]
    
    plt.plot(before['days_from_cutoff'], np.poly1d(np.polyfit(before['days_from_cutoff'], before['Voice_Revenue'], 1))(before['days_from_cutoff']), color='red')
    plt.plot(after['days_from_cutoff'], np.poly1d(np.polyfit(after['days_from_cutoff'], after['Voice_Revenue'], 1))(after['days_from_cutoff']), color='red')
    
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title(f'Regression Discontinuity at {cutoff_date}')
    plt.xlabel('Days from cutoff')
    plt.ylabel('Voice Revenue')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf


    
def run_single_date_rd(df: pd.DataFrame, cutoff_date: str, bandwidth: int = 30) -> Tuple[float, float, float, io.BytesIO]:
    """
    Run regression discontinuity analysis for a single date.
    
    :param df: Input DataFrame
    :param cutoff_date: The date of the intervention/cutoff
    :param bandwidth: Number of days before and after the cutoff to include
    :return: Tuple of (effect, standard_error, p_value, plot_buffer)
    """
    prepared_df = prepare_data_for_rd(df, cutoff_date)
    effect, se, p_value = run_rd_analysis(prepared_df, bandwidth)
    plot_buffer = plot_rd(prepared_df, cutoff_date, bandwidth)
    return effect, se, p_value, plot_buffer
    
    
    
def run_multiple_dates_rd(df: pd.DataFrame, cutoff_dates: List[str], bandwidth: int = 30) -> Tuple[pd.DataFrame, List[io.BytesIO]]:
    results = []
    plot_buffers = []
    for date in cutoff_dates:
        date_obj = pd.to_datetime(date).date()
        effect, se, p_value, plot_buffer = run_single_date_rd(df, str(date_obj), bandwidth)
        results.append({
            'cutoff_date': date_obj,
            'effect': effect,
            'standard_error': se,
            'p_value': p_value
        })
        plot_buffers.append(plot_buffer)
    return pd.DataFrame(results), plot_buffers


def process_uploaded_file(df):
    # st.dataframe(df)
    # st.write(df.columns)
    df['date'] = pd.to_datetime(df['id_date']).dt.normalize()
    df = df.drop(columns=['id_date'])
    df = df.sort_values(by='date')
    df['realized_rate'] = (df['Off_Net'] + df['On_Net']) / df['Voice_Revenue']
    # st.write(df.columns)
    # st.write("DataFrame after processing:")
    # st.write(df.head())
    return df
    
    
@st.cache_data
def load_forecast_data():
    try:
        forecast_df = pd.read_csv("./voice_forecast.csv")
        # forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        return forecast_df
    except FileNotFoundError:
        st.error("voice_forecast.csv not found. Please ensure the file is in the correct directory.")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("./struct_data1.csv")
        # st.write("DataFrame after initial load:")
        # st.write(df.head())
        
        results_df = pd.read_csv("./results_df.csv")
        # st.write("Results DataFrame:")
        # st.write(results_df.head())
        
        processed_df = process_uploaded_file(df)
        
        return processed_df, results_df 
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}. Please contact the admin to upload and process the data.")
        return None, None
        
    
def initialize_session_state():
    if 'df' not in st.session_state or 'results_df' not in st.session_state or 'forecast_df' not in st.session_state:
        df, results_df = load_data()
        forecast_df = load_forecast_data()
        if df is not None and results_df is not None and forecast_df is not None:
            st.session_state.df = df
            st.session_state.results_df = results_df
            st.session_state.forecast_df = forecast_df
            st.session_state.columns = list(df.columns)
        else:
            st.session_state.df = pd.DataFrame()
            st.session_state.results_df = pd.DataFrame()
            st.session_state.forecast_df = pd.DataFrame()
            st.session_state.columns = []

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ""
        
def extract_python_code(text):
    # First, try to find Python code blocks enclosed in triple backticks
    python_blocks = re.findall(r'```python\s*([\s\S]*?)```', text, re.MULTILINE)
    
    if python_blocks:
        return python_blocks
    
    # If no triple backtick blocks found, extract individual lines of code
    lines = text.split('\n')
    code_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    
    return code_lines


def is_plot_code(code):
    return 'plt.' in code or 'matplotlib' in code

def execute_and_save_plot(code, df):
    buf = io.BytesIO()
    exec_globals = {
        'df': df,
        'plt': plt,
        'pd': pd,
        'np': np
    }
    exec(code, exec_globals)
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    if os.path.exists('plot.png'):
        os.remove('plot.png')
    img.save('plot.png')
    return 'plot.png'

def get_llm_response(prompt, chat_history, columns) -> tuple:
    system_prompt = f"""You are an expert data analysis assistant specializing in Python, pandas, and matplotlib. Generate Python code to analyze a DataFrame `df` with the following structure: 
    - **Columns**: {columns} 
    Here is the description of the columns used:
    {{
        'date': 'Date of the recorded data.',
        # Revenue-related metrics
        'Voice_Revenue': 'Total revenue generated from voice services.',
        'recharge': 'Total recharge amount by users.',
        'NEO_VOICE': 'Revenue from voice services under the NEO package.',
        'TUNUKIWA_NEO_VOICE': 'Revenue from NEO voice services under the Tunukiwa offer.',
        'TUNUKIWA_VOICE': 'Revenue from voice services under the Tunukiwa offer.',
        'VOICE_OKOA': 'Revenue from Okoa Jahazi service for voice.',
        'VOICE_ROAMING': 'Revenue from voice services used while roaming.',
        # Minutes
        'Off_Net': 'Minutes of voice traffic to other networks.',
        'On_Net': 'Minutes of voice traffic within the same network.',
        # User counts and service usage
        'Charged_Voice_Users_Pre_Paid': 'Number of prepaid users charged for voice services.',
        'Messaging_Users_Pre_Paid': 'Number of prepaid users using messaging services.',
        'Okoa_Jahazi_Internet_Users_Pre_Paid': 'Number of prepaid users using Okoa Jahazi for internet.',
        'One_Month_Active_Pre_Paid': 'Number of prepaid users active in the last month.',
        'One_Month_Active_Unknown': 'Number of users with unknown status active in the last month.',
        'Prior_Activations_Pre_Paid': 'Previous activations for prepaid users.',
        'SMS_P2P_Customers_Pre_Paid': 'Number of prepaid customers using SMS peer-to-peer services.',
        'Skiza_Users_Pre_Paid': 'Number of prepaid users subscribed to Skiza tunes.',
        'Smartphone_Users_Pre_Paid': 'Number of prepaid users with smartphones.',
        'Successful_Rvrs_Callers_Pre_Paid': 'Number of prepaid users successfully reversing calls.',
        'Three_Month_Active_Pre_Paid': 'Number of prepaid users active in the last three months.',
        'Three_Month_Active_Unknown': 'Number of users with unknown status active in the last three months.',
        'Three_Month_Churn_Pre_Paid': 'Number of prepaid users lost in the last three months.',
        'Three_Month_Reconnections_Pre_Paid': 'Number of prepaid users reconnected in the last three months.',
        # Network performance metrics
        'SUM_TCH_Drops_Daily': 'Total number of daily dropped TCH (Traffic Channel) calls.',
        'AVG_TCH_Drops_Daily': 'Average number of daily dropped TCH (Traffic Channel) calls.',
        'AVG_CSSR_Daily': 'Average daily call setup success rate (CSSR).',
        # Claculated vlues
        'realized_rate':'Total (Off_Net + On_Net) divided by Voice_Revenue
        # Binary indicators (yes/no)
        'is_Weekday': 'Indicator if the day is a weekday (1 for Yes, 0 for No).',
        'is_holiday': 'Indicator if the day is a holiday (1 for Yes, 0 for No).',
        'dayofweek': 'Day of the week (0=Monday, 6=Sunday).',
        'weekend': 'Indicator if the day is a weekend (1 for Yes, 0 for No).',
        'dayofyear': 'Day number in the year (1-365).',
        'dayofmonth': 'Day number in the month (1-31).',
        'weekofyear': 'Week number in the year (1-52).',
        'end': 'Indicator if the day is the end of the month (1 for Yes, 0 for No).',
        'weekofmonth': 'Week number in the month (1-5).',
       
    }}
    - **'date'**: Dates formatted as '2023-04-01' and '2024-04-01' for year-over-year comparison 
    - **'Voice_Revenue'**: Revenue figures
    
    ### Guidelines:
    1. **Code Generation**:
    - Provide only the necessary Python code for DataFrame queries, without explanations or comments.
    - Use very precise code and avoid writing many lines of code
    - For any plotting answers, always make sure you sort the dates and convert them to string in the python code
    - Make sure the python code you generate does not have comments
    - In your dataframe filtering use regex=True to make it more robust


    
    2. **Clarifications**:
    - If the user's request is unclear, incomplete, or missing specific details , **prompt them for clarification** before generating code.
    - Continue the conversation only after receiving the necessary details from the user.
    
    3. **Non-DataFrame Questions**:
    - Request specific details about the finance workbook if the user asks a non-DataFrame-related question.
    - Answer general queries normally but encourage finance-specific questions.
    
    4. **Conciseness**:
    - Produce concise, functional Python code for DataFrame analysis and make sure all parenthesis are used correctly and closed.
    
    Respond to greetings apprpriately i.e Hello , I am RAVEN what can I help you with?
    """
    full_prompt = f"{system_prompt}\nUser: {prompt}\nAssistant:"
    response = bedrock_llm.complete(full_prompt)
    logging.info(f"User: {prompt}\nAssistant: {response.text.strip()}")
    
    updated_chat_history = f"{chat_history}\nUser: {prompt}\nAssistant: {response.text.strip()}"
    return response.text.strip(), updated_chat_history


def handle_effect_query(df, date, action, bandwidth=30):
    try:
        date = pd.to_datetime(date)
        if date.date() not in df['date'].dt.date.values:
            return f"Error: The date {date.date()} is not present in the dataset. Please choose a date within the range of the data."
        effect, se, p_value, plot_buffer = run_single_date_rd(df, str(date.date()), bandwidth)
        
        formatted_output = (
            f"Effect of {action} on {date.date()}:\n"
            f"- Effect: {round(effect):,}\n"
            f"- Standard Error: {round(se):,}\n"
            f"- P-value: {p_value:.4f}"
        )
        
        # Convert plot_buffer to a Figure object
        fig = plt.figure(figsize=(12, 6))
        img = plt.imread(plot_buffer)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Plot for {date.date()}")
        
        return formatted_output, fig
    except Exception as e:
        return f"An error occurred while processing the effect query: {str(e)}"

def handle_multiple_effects_query(df, dates, bandwidth=30):
    try:
        results_df, plot_buffers = run_multiple_dates_rd(df, [str(date.date()) for date in dates], bandwidth)
        
        output = []
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            plot_buffer = plot_buffers[i]
            
            formatted_output = (
                f"Effect on {row['cutoff_date']}:\n"
                f"- Effect: {round(row['effect']):,}\n"
                f"- Standard Error: {round(row['standard_error']):,}\n"
                f"- P-value: {row['p_value']:.4f}"
            )
            
            # Convert plot_buffer to a Figure object
            fig = plt.figure(figsize=(12, 6))
            img = plt.imread(plot_buffer)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Plot for {row['cutoff_date']}")
            
            output.append({"text": formatted_output, "plot": fig})
        
        return output
    except Exception as e:
        return f"An error occurred while processing the multiple effects query: {str(e)}"

def handle_cause_query(df, start_date, end_date, adjustment_note=""):
    try:
        # Add the advanced causal analysis
        features = df.columns.tolist()
        features = [f for f in features if f not in ['date', 'days_from_cutoff', 'post_treatment']]
        
        # Step 1: Create the causal graph
        causal_graph = create_causal_graph(features)
        
        # Filter data for the specified date range
        date_range_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Prepare X and y for the causal analysis
        y = date_range_data['Voice_Revenue']
        X = date_range_data[[col for col in date_range_data.columns if col not in ['Voice_Revenue', 'date', 'days_from_cutoff', 'post_treatment', 'sin_day', 'cos_day']]]
        
        # Step 2: Run double ML with weighting
        importances = double_ml_with_weighting(X, y, causal_graph)
        
        # Step 3: Perform sensitivity analysis
        sensitivities = sensitivity_analysis(X, y, causal_graph, n_iterations=20)  # Reduced iterations for Streamlit performance
        
        # Create results dataframe
        causal_results_df = pd.DataFrame({
            'Feature': features,
            'ATE': [importances.get(f, 0) for f in features],
            'Sensitivity': [sensitivities.get(f, 0) for f in features]
        })
        causal_results_df = causal_results_df.sort_values('ATE', key=abs, ascending=False)
        
        # Calculate t-statistic equivalent
        causal_results_df['t_stat'] = causal_results_df['ATE'].abs() / causal_results_df['Sensitivity'].replace(0, 1e-10)
        
        # Continue with the existing visualization code
        df['realized_rate'] = (df['Off_Net'] + df['On_Net']) / df['Voice_Revenue']
        # Determine if it's a single date query
        is_single_date = start_date == end_date
        
        if is_single_date:
            analysis_start_date = analysis_end_date = start_date
            historical_start_date = historical_end_date = start_date - pd.Timedelta(days=1)
            plot_start_date = end_date - pd.Timedelta(days=6)
            plot_end_date = end_date
        else:
            analysis_start_date, analysis_end_date = start_date, end_date
            historical_end_date = start_date - pd.Timedelta(days=1)
            historical_start_date = historical_end_date - (end_date - start_date)
            plot_start_date, plot_end_date = historical_start_date, end_date
        
        # Calculate features and averages for analysis
        numeric_columns = date_range_data.select_dtypes(include=[np.number]).columns
        today_features = date_range_data[numeric_columns].mean()
        
        # Remove 'Voice_Revenue', 'sin_day', and 'cos_day' columns
        columns_to_remove = ['Voice_Revenue', 'sin_day', 'cos_day']
        today_features = today_features.drop(labels=columns_to_remove, errors='ignore')
        
        current_revenue = date_range_data['Voice_Revenue'].sum()
        date_range_length = (analysis_end_date - analysis_start_date).days + 1
        target_revenue = 250000000 * date_range_length
        revenue_increase_needed = target_revenue - current_revenue
        
        # Create and display the plot
        plot_data = df[(df['date'] >= plot_start_date) & (df['date'] <= plot_end_date)]
        fig1, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_data['date'], plot_data['Voice_Revenue'], label='Voice Revenue', color='blue')
        
        if is_single_date:
            ax.scatter(end_date, plot_data.loc[plot_data['date'] == end_date, 'Voice_Revenue'].values[0], 
                       color='red', s=100, zorder=5, label='Specified Date')
            ax.set_title(f'Voice Revenue (Last 7 Days vs Specified Date: {end_date.date()})')
        else:
            ax.axvspan(start_date, end_date, color='red', alpha=0.3, label='Specified Date Range')
            ax.set_title('Voice Revenue Comparison')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Voice Revenue')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Function to check if a feature is seasonal
        def is_seasonal(feature_name):
            seasonal_keywords = ['day', 'week', 'month', 'season', 'year']
            return any(keyword in feature_name.lower() for keyword in seasonal_keywords)
        
        # Apply the seasonal check to our causal results
        causal_results_df['is_seasonal'] = causal_results_df['Feature'].apply(is_seasonal)
        
        # Filter for positive ATEs only for recommendations
        causal_results_df = causal_results_df[causal_results_df['ATE'] > 0]
        
        # Sort by seasonality and ATE
        causal_results_df = causal_results_df.sort_values(['is_seasonal', 'ATE'], ascending=[False, False])
        
        # Get top features
        top_8 = causal_results_df.head(8)
        
        output = [
            f"Voice Revenue ({start_date.date()} to {end_date.date()}): {math.ceil(current_revenue):,}\n",
            f"Target Revenue (for {date_range_length} days): {math.ceil(target_revenue):,}\n",
            f"Revenue Increase Needed: {math.ceil(revenue_increase_needed):,}\n"
        ]
        
        if adjustment_note:
            output.insert(1, f"{adjustment_note}")
        
        output.append("\nTop Features to Increase Voice Revenue (Based on Causal Analysis):")
        
        # Calculate total positive ATE for non-seasonal features
        total_ate_non_seasonal = top_8[~top_8['is_seasonal']]['ATE'].sum()
        percent_changes, feature_names, estimated_increases = [], [], []
        
        output.append("\nSeasonal Features:")
        for i, (_, row) in enumerate(top_8[top_8['is_seasonal']].iterrows(), 1):
            significant = "Significant" if (row['t_stat'] > 1.96) else "Not significant"
            feature_output = [
                f"{i}. {row['Feature']}:\n",
                f"   • ATE: {row['ATE']:,.2f}\n",
                f"   • Sensitivity: {row['Sensitivity']:,.4f}\n",
                f"   • t-statistic: {row['t_stat']:,.2f}\n",
                f"   • {significant}\n",
                "   CVM Recommendations:"
            ]
            
            if 'day' in row['Feature'].lower():
                feature_output.extend([
                    "   - Analyze daily customer behavior patterns",
                    "   - Optimize daily promotions or service offerings",
                    "   - Improve customer engagement strategies for different days of the week"
                ])
            elif 'week' in row['Feature'].lower():
                feature_output.extend([
                    "   - Develop weekly targeted marketing campaigns",
                    "   - Adjust staffing levels based on weekly demand patterns",
                    "   - Create weekly loyalty programs or incentives"
                ])
            elif 'month' in row['Feature'].lower():
                feature_output.extend([
                    "   - Implement monthly themed promotions or events",
                    "   - Adjust product or service offerings based on monthly trends",
                    "   - Develop monthly customer retention strategies"
                ])
            elif 'season' in row['Feature'].lower():
                feature_output.extend([
                    "   - Create seasonal product bundles or service packages",
                    "   - Develop off-season strategies to boost revenue",
                    "   - Implement seasonal pricing strategies"
                ])
            elif 'year' in row['Feature'].lower():
                feature_output.extend([
                    "   - Develop long-term customer loyalty programs",
                    "   - Implement annual customer review and engagement initiatives",
                    "   - Create yearly milestones or rewards for customer retention"
                ])
            
            feature_output.append("")
            output.extend(feature_output)
        
        output.append("\nNon-Seasonal Features:")
        for i, (_, row) in enumerate(top_8[~top_8['is_seasonal']].iterrows(), 1):
            significant = "Significant" if (row['t_stat'] > 1.96) else "Not significant"
            proportion_of_total_ate = row['ATE'] / total_ate_non_seasonal
            current_value = today_features[row['Feature']]*date_range_length
            new_value = current_value + row['ATE']*date_range_length
            percent_change = ((row['ATE']*date_range_length) / (current_value*date_range_length)) * 100
            estimated_increase = row['ATE']*date_range_length
            Increase_by = new_value-current_value
        
            feature_output = [
                f"{i}. {row['Feature']}:\n",
                f"   • ATE: {row['ATE']:,.2f}\n",
                f"   • Sensitivity: {row['Sensitivity']:,.4f}\n",
                f"   • t-statistic: {row['t_stat']:,.2f}\n",
                f"   • {significant}\n",
                f"   • Current value: {current_value:,.2f} total for {date_range_length} days\n",
                f"   • Recommended new value: {new_value:,.2f} total for {date_range_length} days\n",
                f"   • Increase by: {Increase_by:,.2f}\n",
                f"   • Percent increase needed: {percent_change:.2f}%\n",
                ""
            ]
            
            output.extend(feature_output)
            percent_changes.append(percent_change)
            feature_names.append(row['Feature'])
            estimated_increases.append(estimated_increase)
        
        # Visualize top non-seasonal features with confidence intervals based on sensitivity
        non_seasonal_top = top_8[~top_8['is_seasonal']]
        fig2, ax = plt.subplots(figsize=(12, 8))
        ax.errorbar(non_seasonal_top['Feature'], 
                    non_seasonal_top['ATE'], 
                    yerr=non_seasonal_top['Sensitivity'] * 1.96,  # 95% confidence interval
                    fmt='o', capsize=5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title('Top Non-Seasonal Features: Average Treatment Effects with 95% Confidence Intervals')
        ax.set_xlabel('Feature')
        ax.set_ylabel('ATE')
        plt.xticks(rotation=45, ha='right')
        
        for i, row in enumerate(non_seasonal_top.iterrows()):
            significant = row[1]['t_stat'] > 1.96
            color = 'green' if significant else 'orange'
            ax.plot(i, row[1]['ATE'], 'o', color=color, markersize=10)
        
        ax.legend(['t-stat > 1.96 (Significant)', 't-stat <= 1.96 (Not Significant)'], loc='upper right')
        plt.tight_layout()
        
        # Visualize recommended percent changes for non-seasonal features
        fig3, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(feature_names, percent_changes)
        ax.set_title('Recommended Percent Increases for Non-Seasonal Features')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Percent Increase Needed')
        plt.xticks(rotation=45, ha='right')
        
        for i, (bar, increase) in enumerate(zip(bars, estimated_increases)):
            ax.text(i, bar.get_height(), f'{increase:,.0f}', 
                    ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        total_estimated_increase = sum(estimated_increases)
        output.extend([
            f"\nTotal estimated revenue increase from non-seasonal features: {total_estimated_increase:,.0f}",
            f"\nDifference from target increase: {total_estimated_increase - revenue_increase_needed:,.0f}",
            "\nNote: The impact of seasonal features is not included in this total. Their effect should be addressed through the CVM strategies outlined above."
        ])
        
        output_text = '\n'.join(output)
        
        # Add this at the end of the function
        columns_to_plot = [
            'realized_rate','Charged_Voice_Users_Pre_Paid', 'Messaging_Users_Pre_Paid',
            'Okoa_Jahazi_Internet_Users_Pre_Paid', 'One_Month_Active_Pre_Paid',
            'One_Month_Active_Unknown', 'Prior_Activations_Pre_Paid',
            'SMS_P2P_Customers_Pre_Paid', 'Skiza_Users_Pre_Paid',
            'Smartphone_Users_Pre_Paid', 'Successful_Rvrs_Callers_Pre_Paid',
            'Three_Month_Active_Pre_Paid', 'Three_Month_Active_Unknown',
            'Three_Month_Churn_Pre_Paid', 'Three_Month_Reconnections_Pre_Paid',
            'Off_Net', 'On_Net', 'recharge', 'freq',
            'SUM_TCH_Drops_Daily', 'AVG_TCH_Drops_Daily', 'AVG_CSSR_Daily',
            'NEO_VOICE', 'TUNUKIWA_NEO_VOICE', 'TUNUKIWA_VOICE', 'VOICE_OKOA',
            'VOICE_ROAMING'
        ]
        
        if start_date == end_date:
            additional_data = df[df['date'] == start_date][columns_to_plot].to_dict('records')[0]
        else:
            additional_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)][['date'] + columns_to_plot]
        
        return {
            "revenue_plot": fig1,
            "text_output": output_text,
            "additional_plots": [fig2, fig3],
            "additional_data": additional_data,
            "is_single_date": start_date == end_date,
            "causal_results": causal_results_df  # Adding the causal results for potential further analysis
        }

    except Exception as e:
        return f"An error occurred while processing the cause query: {str(e)}"
    

def create_interactive_plot(df):
    fig = make_subplots(rows=len(df.columns)-1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    for i, column in enumerate(df.columns[1:], 1):  # Skip the 'date' column
        fig.add_trace(go.Scatter(x=df['date'], y=df[column], name=column), row=i, col=1)
        fig.update_yaxes(title_text=column, row=i, col=1)
    
    fig.update_layout(height=200*len(df.columns), showlegend=False, title_text="Interactive Column Values Over Time")
    return fig

def format_single_date_data(data):
    return {k: f"{v:,.2f}" if isinstance(v, (int, float)) else v for k, v in data.items()}
    

def parse_date_range(text):
    date_matches = re.findall(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', text)
    if len(date_matches) == 2:
        try:
            start_date = pd.to_datetime(date_matches[0])
            
            # For end_date, use a custom parsing to handle invalid days
            year, month, day = map(int, re.split(r'[-/]', date_matches[1]))
            last_day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
            end_date = pd.Timestamp(year, month, min(day, last_day.day))
            
            original_end_date = pd.to_datetime(date_matches[1], format='%Y-%m-%d', errors='coerce')
            
            # Check if the end date was adjusted and create a notification
            date_adjusted = end_date != original_end_date
            adjustment_note = f"Note: Adjusted end date from {date_matches[1]} to {end_date.date()} (last day of the month)" if date_adjusted else ""
            
            return sorted([start_date, end_date]), adjustment_note
        except ValueError as e:
            return None, f"Error parsing dates: {str(e)}"
    elif len(date_matches) == 1:
        try:
            date = pd.to_datetime(date_matches[0])
            return [date, date], ""
        except ValueError as e:
            return None, f"Error parsing date: {str(e)}"
    else:
        return None, "No valid dates found in the input."
        
        
def forecast_function(start_date, end_date=None):
    forecast_df = st.session_state.forecast_df
    years = ['2024', '2023', '2022', '2021']  # Reordered years
    
    if end_date is None:
        # Single day forecast
        forecast_data = forecast_df[forecast_df['dayofyear'] == start_date.timetuple().tm_yday]
        values = [forecast_data[year].values[0] if not forecast_data[year].empty else None for year in years]
        
        result_df = pd.DataFrame({
            'Year': years,
            'Forecast Value': values
        })
        result_df['Forecast Value'] = result_df['Forecast Value'].apply(lambda x: f"{float(x):,.2f}" if pd.notnull(x) else "No data")
        
        return result_df, None  # No plot for single day
    else:
        # Date range forecast
        start_day = start_date.timetuple().tm_yday
        end_day = end_date.timetuple().tm_yday
        
        filtered_df = forecast_df[(forecast_df['dayofyear'] >= start_day) & (forecast_df['dayofyear'] <= end_day)].sort_values('dayofyear')
        
        date_labels = [datetime(2024, 1, 1) + timedelta(days=day-1) for day in filtered_df['dayofyear']]
        
        fig = go.Figure()
        colors = ['purple', 'red', 'green', 'blue']  # Reordered colors to match years
        
        for year, color in zip(years, colors):
            fig.add_trace(go.Scatter(
                x=date_labels,
                y=filtered_df[year].astype(float),
                mode='lines+markers',
                name=year,
                line=dict(color=color),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f'Interactive Forecast Comparison: {start_date.date()} to {end_date.date()}',
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Year',
            hovermode='x unified'
        )
        
        fig.update_xaxes(
            tickformat='%Y-%m-%d',
            tickangle=45,
            tickmode='auto',
            nticks=20
        )
        
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        result_data = []
        for year in years:
            year_data = filtered_df[year].astype(float)
            result_data.append({
                'Year': year,
                'Min Value': f"{year_data.min():,.2f}",
                'Max Value': f"{year_data.max():,.2f}",
                'Mean Value': f"{year_data.mean():,.2f}",
                'Sum Value': f"{year_data.sum():,.2f}"
            })
        
        result_df = pd.DataFrame(result_data)
        
        return result_df, fig


import re
from fuzzywuzzy import fuzz

def fuzzy_match(text, keywords, threshold=70):
    return any(fuzz.partial_ratio(keyword, text.lower()) >= threshold for keyword in keywords)
        

def main_logic(text, chat_history, columns, df):
    try:
        date_column = 'date'
        
        if fuzzy_match(text, ['cause', 'recommender', 'explain', 'why']):
            dates, adjustment_note = parse_date_range(text)
            if dates:
                start_date, end_date = dates
                result = handle_cause_query(df, start_date, end_date, adjustment_note)
                # The result now contains all the necessary information, including additional data
                code = f"handle_cause_query(df, pd.to_datetime('{start_date.date()}'), pd.to_datetime('{end_date.date()}'), '{adjustment_note}')"
                return result, chat_history, code
            else:
                output = adjustment_note or "Please provide a specific date range in the format YYYY-MM-DD to YYYY-MM-DD."
                code = ""
                return output, chat_history, ""
                
        elif fuzzy_match(text, ['Safaricom online']):
            st.write("This will take approx 3 mins to scrape data from the internet")
            result=saf_crew.kickoff().raw
            code=""
            return result, chat_history, ""
            
        elif fuzzy_match(text, ['airtel online', 'competition online']):
            st.write("This will take approx 3 mins to scrape data from the internet")
            result=airtel_crew.kickoff().raw
            code=""
            return result, chat_history, ""
            
        elif fuzzy_match(text, ['compare competition', 'safaricom vs airtel','safaricom to airtel']):
            st.write("This will take approx 3 mins to scrape data from the internet")
        
            saf_result=saf_crew.kickoff()
            Airtel_result=airtel_crew.kickoff()
            result=bedrock_llm.complete(f'compare {saf_result} to {Airtel_result} \
            and share main differences as well as what startegy Safaricom can use to get Airtel Voice customers').text
            code=""
            return result, chat_history, ""
            
        elif fuzzy_match(text, ['negative about safaricom online', 'negative sentiments on safaricom']):
            st.write("This will take approx 3 mins to scrape data from the internet")
            result=saf_online_today(f"What are people saying about Safaricom today {datetime.today().strftime('%Y-%m-%d')}")
            code=""
            return result, chat_history, ""
                
        elif fuzzy_match(text, ['effect', 'impact']):
            dates = [pd.to_datetime(d) for d in re.findall(r'\d{4}-\d{2}-\d{2}', text)]
            if len(dates) == 1:
                date = dates[0]
                action_match = re.search(r'effect of (.*?) on', text, re.IGNORECASE)
                action = action_match.group(1) if action_match else "the event"
                output, fig = handle_effect_query(df, date, action)
                code = f"handle_effect_query(df, '{date.date()}', '{action}')"
                return (output, fig), chat_history, code
            elif len(dates) > 1:
                result = handle_multiple_effects_query(df, dates)
                code = f"handle_multiple_effects_query(df, {[d.date() for d in dates]})"
                return result, chat_history, code
            else:
                output = "Please provide specific date(s) in the format YYYY-MM-DD."
                code = ""
                return output, chat_history, code
                
        elif fuzzy_match(text, ['forecast', 'predict', 'projection']):
            dates = [pd.to_datetime(d) for d in re.findall(r'\d{4}-\d{2}-\d{2}', text)]
            if len(dates) == 1:
                result_df, _ = forecast_function(dates[0])
                output = f"Forecast for {dates[0].date()}:"
                # Return as a tuple to match the expected structure
                return (output, result_df), chat_history, ""
            elif len(dates) == 2:
                start_date, end_date = sorted(dates)
                result_df, plot = forecast_function(start_date, end_date)
                output = f"Forecast for {start_date.date()} to {end_date.date()}:"
                # Return as a tuple to match the expected structure
                return (output, result_df, plot), chat_history, ""
            else:
                output = "Please provide one or two specific dates in the format YYYY-MM-DD for the forecast."
                return output, chat_history, ""
        else:
            response, updated_history = get_llm_response(text, chat_history, columns)
            code_blocks = extract_python_code(response)
            # st.write("Extracted Code Blocks:")
            for i, block in enumerate(code_blocks, 1):
                st.code(f"Block {i}:\n{block}", language="python")
            if len(code_blocks) == 0:
                output = response
            else:
                results = []
                for code_block in code_blocks:
                    if is_plot_code(code_block):
                        results.append(execute_and_save_plot(code_block, df))
                    else:
                        try:
                            result = eval(code_block, {'df': df, 'pd': pd, 'np': np, 'results_df': st.session_state.results_df})
                            if result is not None:
                                results.append(result)
                        except:
                            try:
                                local_vars = {'df': df, 'pd': pd, 'np': np, 'results_df': st.session_state.results_df}
                                global_vars = globals().copy()
                                exec(code_block, global_vars, local_vars)
                                del local_vars['df']
                                del local_vars['results_df']
                                results.extend(local_vars.values())
                            except Exception as e:
                                results.append(f"Error executing code: {str(e)}")
                output = results[0] if len(results) == 1 else results if len(results) > 1 else "No valid output generated from the code."
                code = "\n".join(code_blocks)
        return output, chat_history, code
    except Exception as e:
        return f"Error: {str(e)}, try again", chat_history, ""


def display_chat_content(content):
    if isinstance(content, dict) and "revenue_plot" in content and "text_output" in content and "additional_plots" in content:
        # Handle cause query result
        st.pyplot(content["revenue_plot"])
        st.write(content["text_output"])
        
        with st.expander("View Additional Analysis Plots"):
            for plot in content["additional_plots"]:
                st.pyplot(plot)
        
        # Display causal results if available
        if "causal_results" in content:
            with st.expander("View Detailed Causal Analysis Results"):
                st.dataframe(content["causal_results"])
        
        # New additions for interactive plot and single-date display
        if content["is_single_date"]:
            with st.expander("View Additional Metrics for Single Date"):
                st.json(format_single_date_data(content["additional_data"]))
        else:
            with st.expander("Interactive Column Values Plot"):
                st.plotly_chart(create_interactive_plot(content["additional_data"]))

    elif isinstance(content, list) and all(isinstance(item, dict) and "text" in item and "plot" in item for item in content):
        # Handle multiple effects query result
        for item in content:
            st.write(item["text"])
            st.pyplot(item["plot"])
            st.write("---")  # Add a separator between date results
    elif isinstance(content, tuple):
        if len(content) == 2:  # For single date forecast queries or single effect queries
            output, fig = content
            st.write(output)
            if isinstance(fig, pd.DataFrame):
                st.dataframe(fig, hide_index=True)
            else:
                st.pyplot(fig)
        elif len(content) == 3:  # For date range forecast queries
            output, result_df, plot = content
            st.write(output)
            st.dataframe(result_df, hide_index=True)
            if plot:
                st.plotly_chart(plot)
    else:
        st.write(content)


def main():
    st.title("RAVEN: Revenue Analysis via Event-aware Neural-causal-explainer")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm RAVEN (Revenue Analysis via Event-aware Neural-causal-explainer). How can I help you analyze voice revenue today?"}
        ]
  
    # Define your column descriptions
    column_descriptions = {
        'date': 'Date of the recorded data.',
        'Voice_Revenue': 'Total revenue generated from voice services.',
        'recharge': 'Total recharge amount by users.',
        'NEO_VOICE': 'Revenue from voice services under the NEO package.',
        'TUNUKIWA_NEO_VOICE': 'Revenue from NEO voice services under the Tunukiwa offer.',
        'TUNUKIWA_VOICE': 'Revenue from voice services under the Tunukiwa offer.',
        'VOICE_OKOA': 'Revenue from Okoa Jahazi service for voice.',
        'VOICE_ROAMING': 'Revenue from voice services used while roaming.',
        'Off_Net': 'Minutes of voice traffic to other networks.',
        'On_Net': 'Minutes of voice traffic within the same network.',
        'Charged_Voice_Users_Pre_Paid': 'Number of prepaid users charged for voice services.',
        'Messaging_Users_Pre_Paid': 'Number of prepaid users using messaging services.',
        'Okoa_Jahazi_Internet_Users_Pre_Paid': 'Number of prepaid users using Okoa Jahazi for internet.',
        'One_Month_Active_Pre_Paid': 'Number of prepaid users active in the last month.',
        'One_Month_Active_Unknown': 'Number of users with unknown status active in the last month.',
        'Prior_Activations_Pre_Paid': 'Previous activations for prepaid users.',
        'SMS_P2P_Customers_Pre_Paid': 'Number of prepaid customers using SMS peer-to-peer services.',
        'Skiza_Users_Pre_Paid': 'Number of prepaid users subscribed to Skiza tunes.',
        'Smartphone_Users_Pre_Paid': 'Number of prepaid users with smartphones.',
        'Successful_Rvrs_Callers_Pre_Paid': 'Number of prepaid users successfully reversing calls.',
        'Three_Month_Active_Pre_Paid': 'Number of prepaid users active in the last three months.',
        'Three_Month_Active_Unknown': 'Number of users with unknown status active in the last three months.',
        'Three_Month_Churn_Pre_Paid': 'Number of prepaid users lost in the last three months.',
        'Three_Month_Reconnections_Pre_Paid': 'Number of prepaid users reconnected in the last three months.',
        'SUM_TCH_Drops_Daily': 'Total number of daily dropped TCH (Traffic Channel) calls.',
        'AVG_TCH_Drops_Daily': 'Average number of daily dropped TCH (Traffic Channel) calls.',
        'AVG_CSSR_Daily': 'Average daily call setup success rate (CSSR).',
        'realized_rate': 'Total (Off_Net + On_Net) divided by Voice_Revenue',
        'is_Weekday': 'Indicator if the day is a weekday (1 for Yes, 0 for No).',
        'is_holiday': 'Indicator if the day is a holiday (1 for Yes, 0 for No).',
        'dayofweek': 'Day of the week (0=Monday, 6=Sunday).',
        'weekend': 'Indicator if the day is a weekend (1 for Yes, 0 for No).',
        'dayofyear': 'Day number in the year (1-365).',
        'dayofmonth': 'Day number in the month (1-31).',
        'weekofyear': 'Week number in the year (1-52).',
        'end': 'Indicator if the day is the end of the month (1 for Yes, 0 for No).',
        'weekofmonth': 'Week number in the month (1-5).',
    }
    
    # Sidebar
    with st.sidebar:
        st.title("Welcome to RAVEN")
        st.info(
            "This app is an LLM-powered voice causal agent with event detection and forecasting capabilities.\n\n"
            "Example prompts:\n"
            "- What was the **cause** of the voice revenue change on YYYY-MM-DD or between YYYY-MM-DD and YYYY-MM-DD?\n"
            "- **Explain** or **recommend** actions for the voice revenue on YYYY-MM-DD or for the period YYYY-MM-DD to YYYY-MM-DD.\n"
            "- What was the **effect** of action X on the voice revenue on YYYY-MM-DD?\n"
            "- What were the **effects** on voice revenue on YYYY-MM-DD, YYYY-MM-DD, and YYYY-MM-DD?\n"
            "- **Forecast** the voice revenue for YYYY-MM-DD or from YYYY-MM-DD to YYYY-MM-DD.\n"
            "- **ONLINE ANALYSIS** What is the comparison between Safaricom and competition.\n"
            "- What are some Voice product views for Airtel online.\n"
            "- What are some Voice product views for Safaricom online.\n"
            "- What negative sentiments on Safaricom exist online today?\n"
        )
        st.markdown("---")
        with st.expander("Column Descriptions"):
            for column, description in column_descriptions.items():
                st.markdown(f"**{column}**: {description}")
        st.markdown("---")
        st.markdown("""
        ### Understanding Causal Analysis

        **Average Treatment Effect (ATE)**: 
        - Measures the causal impact of changing a feature on voice revenue
        - Positive value: Increasing this feature increases revenue
        - Negative value: Increasing this feature decreases revenue
        
        **Sensitivity**:
        - Measures uncertainty in the causal estimate
        - Lower values indicate more reliable estimates
        
        **t-statistic**:
        - Ratio of ATE to Sensitivity
        - Values > 1.96 suggest statistically significant effects (95% confidence)

        ### Understanding Event Effects

        - **Effect**: Estimated change in voice revenue due to the event.
          - Positive: Revenue increased
          - Negative: Revenue decreased
    
        - **Standard Error**: Indicates the reliability of the effect estimate.
          - If much smaller than Effect: Estimate is more reliable
          - If close to or larger than Effect: Estimate is less reliable
          - Rule of thumb: Look for Effect at least 2 times larger than Standard Error
    
        - **P-value**: Indicates how likely the effect is due to chance.
          - < 0.05: Strong evidence the event caused the change
          - 0.05 - 0.10: Moderate evidence the event caused the change
          - >0.10: Weak evidence; the change might be due to chance
        """)
        
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            display_chat_content(message["content"])

    # User input
    user_input = st.chat_input("Enter your question about the voice revenue data:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    result, _, code = main_logic(
                        user_input,
                        "",  # We're not using chat_history in the old format
                        st.session_state.columns,
                        st.session_state.df
                    )

                    # Add assistant's response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    display_chat_content(result)

                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.write(error_message)

if __name__ == "__main__":
    initialize_session_state()
    if (not st.session_state.df.empty and 
        not st.session_state.results_df.empty and 
        not st.session_state.forecast_df.empty):
        main()
    else:
        if st.session_state.forecast_df.empty:
            st.warning("Forecast data not available. Please ensure voice_forecast.csv is present in the directory.")
        else:
            st.warning("No data available. Please ensure struct_data1.csv, results_df.csv, and voice_forecast.csv are present in the directory.")