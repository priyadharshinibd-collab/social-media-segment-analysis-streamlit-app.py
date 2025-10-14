import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Sentiment Data Viewer and Dashboard')

st.write("This app displays the first few rows of the sentiment dataset and includes some visualizations.")

# Load the data (adjust the path if necessary)
file_path = '/content/sentimentdataset.csv'
try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    st.write("Data loaded successfully:")
    st.dataframe(df.head())

    # Ensure Timestamp is datetime and Year_Month is created
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp']) # Drop rows with invalid timestamps
        df['Year_Month'] = df['Timestamp'].dt.to_period('M')


    # --- Visualization 1: Sentiment Distribution ---
    st.subheader("Distribution of Sentiment Labels")
    if 'Sentiment' in df.columns:
        plt.figure(figsize=(10, 8))
        top_n_sentiments = 15
        top_sentiments = df['Sentiment'].value_counts().nlargest(top_n_sentiments).index
        sns.countplot(data=df[df['Sentiment'].isin(top_sentiments)],
                      y='Sentiment', order=top_sentiments, palette='viridis')
        plt.title(f'Distribution of Top {top_n_sentiments} Sentiment Labels')
        plt.xlabel('Count')
        plt.ylabel('Sentiment')
        st.pyplot(plt) # Use st.pyplot() to display the plot in Streamlit
        plt.close() # Close the plot to prevent displaying it twice

    # --- Visualization 2: Sentiment Over Time (Monthly) ---
    st.subheader("Monthly Sentiment Distribution Over Time")
    if 'Sentiment' in df.columns and 'Year_Month' in df.columns:
        sentiment_trends_monthly = df.groupby(['Year_Month', 'Sentiment']).size().unstack(fill_value=0)
        if not sentiment_trends_monthly.empty:
            fig, ax = plt.subplots(figsize=(15, 8))
            sentiment_trends_monthly.plot(kind='bar', stacked=True, ax=ax)
            plt.title('Monthly Sentiment Distribution Over Time')
            plt.xlabel('Month')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("No monthly sentiment data to visualize.")

    # --- Visualization 3: Geographic Distribution (Sentiment by Country) ---
    st.subheader("Sentiment Distribution by Country")
    if 'Sentiment' in df.columns and 'Country' in df.columns:
        sentiment_by_country = df.groupby(['Country', 'Sentiment']).size().unstack(fill_value=0)
        top_n_countries = 15
        top_countries = df['Country'].value_counts().nlargest(top_n_countries).index
        sentiment_by_country_top = sentiment_by_country.loc[top_countries]

        if not sentiment_by_country_top.empty:
            fig, ax = plt.subplots(figsize=(15, 8))
            sentiment_by_country_top.plot(kind='bar', stacked=True, ax=ax)
            plt.title(f'Sentiment Distribution by Top {top_n_countries} Countries')
            plt.xlabel('Country')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
             st.write("No country sentiment data to visualize.")

    # --- Visualization 4: Historical Sentiment Proportions ---
    st.subheader("Monthly Proportion of Top Sentiments Over Time")
    if 'Sentiment' in df.columns and 'Year_Month' in df.columns:
        top_n_sentiments = 15
        top_sentiments = df['Sentiment'].value_counts().nlargest(top_n_sentiments).index
        df_top_sentiments = df[df['Sentiment'].isin(top_sentiments)]
        sentiment_proportions_monthly = df_top_sentiments.groupby('Year_Month')['Sentiment'].value_counts(normalize=True).unstack(fill_value=0)

        if not sentiment_proportions_monthly.empty:
            fig, ax = plt.subplots(figsize=(15, 8))
            sentiment_proportions_monthly.plot(kind='area', stacked=True, ax=ax)
            plt.title(f'Monthly Proportion of Top {top_n_sentiments} Sentiments Over Time')
            plt.xlabel('Month')
            plt.ylabel('Proportion')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("No monthly sentiment data to visualize proportions.")

    # --- Visualization 5: Platform Distribution ---
    st.subheader("Distribution of Platforms")
    if 'Platform' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, y='Platform', order=df['Platform'].value_counts().index, palette='viridis')
        plt.title('Distribution of Platforms')
        plt.xlabel('Count')
        plt.ylabel('Platform')
        st.pyplot(plt)
        plt.close()

    # --- Visualization 6: Average Retweets and Likes by Sentiment ---
    st.subheader("Average Retweets and Likes by Sentiment")
    if 'Sentiment' in df.columns and ('Retweets' in df.columns or 'Likes' in df.columns):
        sentiment_engagement = df.groupby('Sentiment')[['Retweets', 'Likes']].mean().reset_index()
        sentiment_engagement = sentiment_engagement.melt('Sentiment', var_name='Engagement Type', value_name='Average Count')

        plt.figure(figsize=(12, 8))
        sns.barplot(data=sentiment_engagement, x='Sentiment', y='Average Count', hue='Engagement Type', palette='viridis')
        plt.title('Average Retweets and Likes by Sentiment')
        plt.xlabel('Sentiment')
        plt.ylabel('Average Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()


except FileNotFoundError:
    st.error(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    st.error(f"An error occurred during data loading or visualization: {e}")
