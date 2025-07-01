import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from groq import Groq
import requests
import json
from typing import Dict, List, Optional

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

if 'analyses' not in st.session_state:
    st.session_state.analyses = []

def extract_json_from_response(content: str) -> dict:
    import re
    # Try to find ```json ... ``` block
    match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
    if not match:
        # Try to find generic ``` ... ``` block
        match = re.search(r"```\s*(.*?)```", content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = content.strip()
    return json.loads(json_str)

@st.cache_data(ttl=3600)
def load_nyse_stocks():
    """Load all NYSE stocks from the GitHub repository"""
    url = "https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json"
    try:
        response = requests.get(url)
        data = response.json()
        
        
        stocks_list = []
        for _, company in data.items():
            stocks_list.append({
                'ticker': company['ticker'],
                'title': company['title'],
                'cik_str': company['cik_str']
            })
        
        stocks_df = pd.DataFrame(stocks_list)
        
        
        stocks_df['sector'] = None
        stocks_df['market_cap'] = None
        stocks_df['volume'] = None
        
        
        for idx, row in stocks_df.head(50).iterrows():
            try:
                stock = yf.Ticker(row['ticker'])
                info = stock.info
                stocks_df.at[idx, 'sector'] = info.get('sector', 'Unknown')
                stocks_df.at[idx, 'market_cap'] = info.get('marketCap', 0)
                stocks_df.at[idx, 'volume'] = info.get('volume', 0)
            except:
                continue
        
        return stocks_df
    except Exception as e:
        st.error(f"Error loading NYSE stocks: {str(e)}")
        return pd.DataFrame()

def get_stock_details(symbol: str) -> Optional[Dict]:
    """Get detailed stock information"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
       
        hist = stock.history(period="1mo")
        
        return {
            'symbol': symbol,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'volume': info.get('volume', 0),
            'price': info.get('currentPrice', 0),
            'pe_ratio': info.get('forwardPE', 'N/A'),
            'description': info.get('longBusinessSummary', 'N/A'),
            'historical_data': hist
        }
    except Exception as e:
        st.error(f"Error fetching stock details for {symbol}: {str(e)}")
        return None

def search_stocks(query: str) -> List[Dict]:
    """Search stocks based on natural language query"""
    try:
        prompt = f"""
You are a financial analysis assistant.

Based on this query: "{query}"

Return relevant stock symbols and why they match.

Output ONLY valid JSON enclosed in a Markdown code block.

Example:
```json
{{
    "matches": [
        {{"symbol": "AAPL", "reason": "Example reason"}}
    ]
}}
``` 
"""
        
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        try:
            results = extract_json_from_response(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Model output parsing failed: {str(e)}. Raw output: {response.choices[0].message.content}")
            return []
        return results['matches']
    except Exception as e:
        st.error(f"Error searching stocks: {str(e)}")
        return []

def process_article(article_text: str, article_title: str) -> Optional[Dict]:
    """Process an article using GROQ"""
    try:
        prompt = f"""
Title: {article_title}
Content: {article_text}

Analyze this article and provide a structured analysis in the following JSON format enclosed in a Markdown code block:

```json
{{
    "companies": ["List of company stock symbols mentioned"],
    "sentiment": "Overall sentiment (positive/negative/neutral)",
    "event_type": "Type of event (earnings/merger/product_launch/partnership/regulatory/other)",
    "summary": "Brief 2-sentence summary of the key points",
    "key_metrics": {{
        "price_impact": "Mentioned stock price changes if any",
        "market_impact": "Potential market impact description"
    }}
}}
```

Return ONLY valid JSON.
"""
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        try:
            analysis = extract_json_from_response(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Model output parsing failed: {str(e)}. Raw output: {response.choices[0].message.content}")
            return None

        analysis['processed_at'] = datetime.now().isoformat()
        analysis['title'] = article_title
        return analysis

    except Exception as e:
        st.error(f"Error processing article: {str(e)}")
        return None

def display_stock_chart(data: pd.DataFrame, symbol: str):
    """Display stock price chart"""
    st.line_chart(data['Close'])
    st.bar_chart(data['Volume'])

def main():
    st.title("Market Research and Analysis System")
    

    stocks_df = load_nyse_stocks()
    
    tab1, tab2, tab3 = st.tabs(["Stock Research", "Article Processing", "Market Analysis"])
    
    with tab1:
        st.header("Stock Research")
        
        
        query = st.text_input("Enter your search query (e.g., 'companies that build data centers' or 'largest tech companies')")
        if query:
            with st.spinner("Searching relevant stocks..."):
                matches = search_stocks(query)
                
                for match in matches:
                    with st.expander(f"{match['symbol']}: {match['reason']}"):
                        details = get_stock_details(match['symbol'])
                        if details:
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Price", f"${details['price']:.2f}")
                            with col2:
                                st.metric("Market Cap", f"${details['market_cap']/1e9:.1f}B")
                            with col3:
                                st.metric("Volume", f"{details['volume']:,}")
                            
                          
                            st.write("**Sector:**", details['sector'])
                            st.write("**P/E Ratio:**", details['pe_ratio'])
                            st.write("**Description:**", details['description'])
                            
                        
                            if 'historical_data' in details:
                                st.subheader("Price History (1 Month)")
                                display_stock_chart(details['historical_data'], details['symbol'])
        

        st.subheader("Filter by Metrics")
        available_sectors = stocks_df[stocks_df['sector'].notna()]['sector'].unique()
        sector = st.selectbox("Sector", ['All'] + list(available_sectors))
        min_market_cap = st.number_input("Minimum Market Cap (Billions)", min_value=0.0)
        
        if sector != 'All' or min_market_cap > 0:
            filtered_stocks = stocks_df[
                (stocks_df['sector'] == sector if sector != 'All' else True) &
                (stocks_df['market_cap'] >= min_market_cap * 1e9)
            ].dropna(subset=['sector', 'market_cap'])
            
            st.write(f"Found {len(filtered_stocks)} matching stocks")
            st.dataframe(filtered_stocks[['ticker', 'title', 'sector', 'market_cap', 'volume']])
    
    with tab2:
        st.header("Article Processing")
        
        with st.form("article_form"):
            title = st.text_input("Article Title")
            body = st.text_area("Article Content", height=200)
            col1, col2 = st.columns(2)
            with col1:
                publisher = st.text_input("Publisher (optional)")
            with col2:
                author = st.text_input("Author (optional)")
            
            if st.form_submit_button("Process Article"):
                if not title or not body:
                    st.error("Please provide both title and content.")
                else:
                    with st.spinner("Processing article..."):
                        analysis = process_article(body, title)
                        
                        if analysis:
                            st.success("Article processed successfully!")
                            
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sentiment", analysis['sentiment'].capitalize())
                            with col2:
                                st.metric("Event Type", analysis['event_type'].replace('_', ' ').title())
                            with col3:
                                st.metric("Companies Mentioned", len(analysis['companies']))
                            
                            st.subheader("Analysis Details")
                            st.write("**Companies Mentioned:**")
                            for company in analysis['companies']:
                                st.write(f"- {company}")
                            
                            st.write("**Summary:**")
                            st.write(analysis['summary'])
                            
                            st.write("**Market Impact:**")
                            st.write(analysis['key_metrics']['market_impact'])
                            
                            
                            st.session_state.analyses.append(analysis)
        
       
        if st.session_state.analyses:
            st.subheader("Recent Analyses")
            for idx, analysis in enumerate(reversed(st.session_state.analyses[-5:])):
                with st.expander(f"Analysis {len(st.session_state.analyses) - idx}: {analysis['title']}"):
                    st.write(f"**Processed at:** {analysis['processed_at']}")
                    st.write(f"**Sentiment:** {analysis['sentiment']}")
                    st.write(f"**Event Type:** {analysis['event_type']}")
                    st.write(f"**Companies:** {', '.join(analysis['companies'])}")
                    st.write(f"**Summary:** {analysis['summary']}")
    
    with tab3:
        st.header("Market Analysis")
        
        if st.session_state.analyses:
            
            sentiments = pd.DataFrame([
                {'sentiment': a['sentiment'], 'event_type': a['event_type']}
                for a in st.session_state.analyses
            ])
            
            
            st.subheader("Sentiment Distribution")
            sentiment_counts = sentiments['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            
            
            st.subheader("Event Type Distribution")
            event_counts = sentiments['event_type'].value_counts()
            st.bar_chart(event_counts)
        else:
            st.info("Process some articles to see market analysis.")

if __name__ == "__main__":
    st.set_page_config(page_title="Financial Analyzer", layout="wide")
    main()
