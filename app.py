import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
import re
from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
genini_api_key = os.getenv("GENINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Smart Property Recommender", layout="centered")

st.header("Smart Property Recommender")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
groq_llm = ChatGroq(model_name="llama3-8b-8192")


# ----------------- Load Dataset -----------------

df = pd.read_excel("Case Study 2 Data.xlsx", sheet_name="Property Data")
df['Price'] = df['Price'].astype(str).str.replace('[\$,]', '', regex=True).str.replace('k', '').astype(float)


# Streamlit inputs
bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=3, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, value=2, step=1)
budget = st.number_input("Budget(eg. $500k)", min_value=100, value=400, max_value=1000,step=50)

user_preference = st.text_input(
    "Describe Your Ideal Property",
    placeholder="e.g., 3-bedroom house near downtown with a garden",
    help="Mention key features like number of rooms, location, budget, etc."
)

#----------------- Filter Dataset -----------------

if st.button("Recomend"):
    def filter_properties(df, bedrooms, bathrooms, price, relax_factor=0.2):
        price_min = price * (1 - relax_factor)
        price_max = price * (1 + relax_factor)
        filtered = df[
            (df['Bedrooms'] == bedrooms) &
            (df['Bathrooms'] == bathrooms) &
            (df['Price'] >= price_min) & (df['Price'] <= price_max)
            ]
        if filtered.empty:
            print("No exact matches found. Relaxing filters...")
            filtered = df[
                (df['Bedrooms'].between(bedrooms - 1, bedrooms + 1)) &
                (df['Bathrooms'].between(bathrooms - 1, bathrooms + 1)) &
                (df['Price'] >= price * 0.7) & (df['Price'] <= price * 1.3)
                ]
        return filtered

    filtered_df=filter_properties(df, bedrooms=bedrooms, bathrooms=bathrooms, price=budget, relax_factor=0.2)

    if filtered_df.empty:
            st.error("No properties found even after relaxing filters. Please try different criteria.")
            st.stop()  # 🔥 This will STOP further execution
    else:
        # ----------------- Create FAISS Vectorstore -----------------
        def create_vectorstore(filtered_df):
            # Iterate over DataFrame rows
            property_docs = [
                Document(
                    page_content=str(row['Qualitative Description']),
                    metadata=row.to_dict()
                )
                for _, row in filtered_df.iterrows()  # Use iterrows() to get rows
            ]
            if not property_docs:
                st.write("Not matching try anotrher prefarance")

            vectorstore = FAISS.from_documents(property_docs, embedding=embedding_model)
            return vectorstore


        vector_store = create_vectorstore(filtered_df)

        base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        compressor = LLMChainExtractor.from_llm(groq_llm)
        compression_retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)


        # ----------------- Create FAISS Vectorstore -----------------
        def create_vectorstore(filtered_df):
            # Iterate over DataFrame rows
            property_docs = [
                Document(
                    page_content=str(row['Qualitative Description']),
                    metadata=row.to_dict()
                )
                for _, row in filtered_df.iterrows()  # Use iterrows() to get rows
            ]
            if not property_docs:
                raise ValueError("No documents to embed!")

            vectorstore = FAISS.from_documents(property_docs, embedding=embedding_model)
            return vectorstore


        vector_store = create_vectorstore(filtered_df)

        #------------------Retrivers------------------------

        base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        compressor = LLMChainExtractor.from_llm(groq_llm)
        compression_retriever = ContextualCompressionRetriever(base_retriever=base_retriever,
                                                               base_compressor=compressor)


        def format_property_info(compressed_results):
            """
            Combines Property ID and page content for each property, separated by spaces.
            """
            formatted_info = ""
            for doc in compressed_results:
                property_id = doc.metadata.get("Property ID")
                page_content = doc.page_content
                formatted_info += f"Property ID: {property_id}\n{page_content}\n\n\n"  # Add clear spaces
            return formatted_info


        # ------------ Pydantic Models ------------

        class PropertyRecommendation(BaseModel):
            properties: list[list[int, str]] = Field(
                description="Unique ID (extracted from meta data) of the matching property based on user's preference,and why that property is choosen nased on user prefarance")


        # ------------ Output Parser ------------

        # For structured Pydantic output
        parser = PydanticOutputParser(pydantic_object=PropertyRecommendation)

        # ------------ Real Estate Matching Prompt ------------

        prompt = PromptTemplate(
            template="""
        You are an expert real estate advisor.

        Goal:
        Match user's preferences carefully with available properties based on lifestyle needs (peaceful living, luxury, eco-friendliness, family-friendly, budget-friendly).

        Instructions:
        - Only select properties genuinely aligned with user expectations.
        - If unclear from property description, assume missing info.
        - Indirect matches can be included only if genuinely valuable.
        - If none match, return an empty list [].
        - Output strictly as a list of lists:
          [property_id (int), short reason (str)]
        - If none match, return a property with "property_id": 0 and reason: "No property found. Try modifying your preferences."

        Format:
        {format_instructions}

        Examples:

        Example 1: (Only one property selected)
        User Preference: "Looking for a peaceful, eco-friendly home, preferably with a garden."
        Available Properties:
        - Property ID 1: "Luxury villa near city center, no mention of garden."
        - Property ID 2: "Cozy home in countryside with organic garden and solar panels."
        - Property ID 3: "Apartment in busy downtown."

        Answer:
        [
          [2, "Countryside eco-friendly home with garden perfectly matches peaceful living."]
        ]

        Example 2: (Multiple properties selected)
        User Preference: "Looking for a budget apartment close to public transport and park access."
        Available Properties:
        - Property ID 10: "Affordable studio apartment near metro and city park."
        - Property ID 11: "Small flat, 10 minutes walk from metro station."
        - Property ID 12: "Luxury penthouse far from any park or public transport."

        Answer:
        [
          [10, "Affordable and near both metro and park, fits budget commuting lifestyle."],
          [11, "Close to metro, partially matching commuting preference."]
        ]

        Example 3: (All properties selected)
        User Preference: "Seeking luxurious, large homes with pools for family leisure."
        Available Properties:
        - Property ID 20: "Luxury mansion with large pool, family-friendly amenities."
        - Property ID 21: "Exclusive villa with private pool and kids' play area."
        - Property ID 22: "Elegant bungalow with indoor swimming pool and garden."

        Answer:
        [
          [20, "Luxury mansion with pool and family amenities matches exactly."],
          [21, "Exclusive villa with pool and kids' area ideal for family leisure."],
          [22, "Bungalow with swimming pool and garden fits luxurious family needs."]
        ]

        Example 4: (No property selected)
        User Preference: "Looking for eco-friendly farmhouses with organic farming facilities."
        Available Properties:
        - Property ID 30: "Downtown condo with gym and spa."
        - Property ID 31: "Business hotel suites near airport."
        - Property ID 32: "City center serviced apartments."
        
        Answer:
        [
          ["0,"No property found. Try modifying your preferences."]
        ]

        Now do the same for:

        Input:
        - User Preference: {user_preference}
        - Available Properties: {property_descriptions}

        Only return the structured output, no extra text.
        """,
            input_variables=["user_preference", "property_descriptions"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )


        #-------------------------------Chain---------------------------------

        parallel_chain = RunnableParallel({
            'user_preference': RunnablePassthrough(),
            'property_descriptions': compression_retriever | RunnableLambda(lambda x: format_property_info(x))
        })

        main_chain = parallel_chain | prompt | groq_llm

        result=main_chain.invoke(user_preference).content

        # Initialize empty lists
        ids = []
        texts = []

        # Find all occurrences of the pattern [number, "text"]
        matches = re.findall(r'\[(\d+),\s*"([^"]+)"\]', result)

        # Loop through the matches and populate ids and texts lists
        for match in matches:
            ids.append(int(match[0]))  # Convert the ID to an integer
            texts.append(match[1])  # Extract the text

        # Output the results

        # Filter DataFrame where Property ID is in the list `ids`
        display_df = df[df['Property ID'].isin(ids)].copy()
        display_df["texts"]=texts

        # Loop through each row in display_df
        for index, row in display_df.iterrows():
            i=1
            st.write(f"### Recommended Property {i}:")
            i=i+1

            # Create a green-colored box for the description

            # Display the property details
            st.write(f"**ID:** {row['Property ID']}")
            st.write(f"**Bedrooms:** {row['Bedrooms']}")
            st.write(f"**Bathrooms:** {row['Bathrooms']}")
            st.write(f"**Price:** {row['Price']}")
            st.write(f"**Living Area (sq ft):** {row['Living Area (sq ft)']}")
            st.write(f"**Property Description:** {row['Qualitative Description']}")

            st.markdown(f"""
            <div style="background-color: #D4EDDA; padding: 20px; border-radius: 10px; border: 1px solid #28A745;">
                <h4 style="margin: 0; color: #28A745;">Description</h4>
                <p style="font-size: 16px; color: #333;">{row['texts']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Add a separator between rows for better clarity
            st.markdown("-----------------------------------------------------------")





