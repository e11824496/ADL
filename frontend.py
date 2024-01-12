import streamlit as st
import pandas as pd
import json
import requests
import matplotlib.pyplot as plt


from dataset_generation import BankStatementProcessor

API_URL = "http://backend:8000/classify"

# streamlit wide mode
st.set_page_config(layout="wide")


def fetch_bankstatement_data(line):
    row = st.session_state.merged_dataset.iloc[line]
    return {
        'amount': row.get('Amount', 0),
        'description': row.get('Description', ''),
        'time': str(row.get('Time', ''))
    }


def make_prediction_request(data):
    try:
        response = requests.post(API_URL, json=data)
        response.raise_for_status()
        return json.loads(response.text).split('/') if \
            response.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction request failed: {e}")
        return None


def merge_and_label_data():
    if st.session_state.merged_dataset is None:
        bsp = BankStatementProcessor(
            st.session_state.bankstatements_file,
            st.session_state.paypal_file
        )
        bsp.create_dataset()
        st.session_state.merged_dataset = bsp.bank_statements

    st.session_state.merged_dataset['category_group'] = ''
    st.session_state.merged_dataset['category_subgroup'] = ''
    for index, row in st.session_state.merged_dataset.iterrows():
        data = fetch_bankstatement_data(index)
        predictions = make_prediction_request(data)
        # add predictions to merged_dataset
        st.session_state.merged_dataset.loc[index, 'category_group'] = \
            predictions[0]
        st.session_state.merged_dataset.loc[index, 'category_subgroup'] = \
            predictions[1]


@st.cache_data
def load_data(file_path) -> pd.DataFrame:
    if file_path:
        data = pd.read_csv(file_path)
        return data
    return None


def main():
    st.session_state.setdefault("bankstatements", None)
    st.session_state.setdefault("paypal", None)
    st.session_state.setdefault("merged_dataset", None)

    # CATEGORIES = load_categories("categories.json")

    st.title("Bank Statement Analysis")

    st.sidebar.file_uploader(
        "Upload your Bank-Statements", type=["csv"], key="bankstatements_file")

    st.sidebar.file_uploader(
        "Upload your Paypal", type=["csv"], key="paypal_file")

    st.sidebar.button(
        "Merge and label data", on_click=merge_and_label_data)

    if st.session_state.merged_dataset is not None:
        st.write(st.session_state.merged_dataset)
        group_counts = st.session_state.\
            merged_dataset['category_group'].\
            value_counts()
        plt.pie(group_counts, labels=group_counts.index, autopct='%1.1f%%')
        plt.axis('equal')
        st.pyplot(plt.gcf())


if __name__ == "__main__":
    main()
