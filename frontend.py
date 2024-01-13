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


@st.cache_data
def load_categories(file_path) -> dict:
    with open(file_path) as f:
        categories = json.load(f)
    return categories


def select_group(category):
    st.session_state.selected_category = category


def plot_pie_chart(df, labels):
    fig, ax = plt.subplots()
    df.groupby(labels)['Amount'].sum().abs().plot(
        kind='pie', y='', ax=ax, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)


def main():
    st.session_state.setdefault("bankstatements", None)
    st.session_state.setdefault("paypal", None)
    st.session_state.setdefault("merged_dataset", None)
    st.session_state.setdefault("selected_category", "ALL")

    CATEGORIES = load_categories("categories.json")

    st.title("Bank Statement Analysis")

    st.sidebar.file_uploader(
        "Upload your Bank-Statements", type=["csv"], key="bankstatements_file")

    st.sidebar.file_uploader(
        "Upload your Paypal", type=["csv"], key="paypal_file")

    st.sidebar.button(
        "Merge and label data", on_click=merge_and_label_data)

    if st.session_state.merged_dataset is not None:
        st.subheader("Category Selection")
        n_cols = len(CATEGORIES)//2
        cols = st.columns(n_cols)
        # Create category buttons
        selected_category = st.session_state.selected_category

        for i, group in enumerate(["ALL"] + list(CATEGORIES.keys())):
            highlight = group == selected_category
            cols[i % n_cols].button(
                group, on_click=select_group, args=(group,),
                use_container_width=True,
                type='primary' if highlight else 'secondary')

        if selected_category == "ALL":
            filterd_df = st.session_state.merged_dataset
            labels = "category_group"
        else:
            filterd_df = st.session_state.merged_dataset[
                st.session_state.merged_dataset['category_group'] ==
                selected_category
            ]
            labels = "category_subgroup"
        st.write(filterd_df)

        # plot pie chart for negative values
        st.subheader("Pie Chart for outgoing payments")
        filterd_df_negative = filterd_df[filterd_df['Amount'] < 0]
        if len(filterd_df_negative) == 0:
            st.write("No data for outgoing payments available")
        else:
            plot_pie_chart(filterd_df_negative, labels)

        # plot pie chart for positive values
        st.subheader("Pie Chart for incoming payments")
        filterd_df_positive = filterd_df[filterd_df['Amount'] > 0]
        if len(filterd_df_positive) == 0:
            st.write("No data for incoming payments available")
        else:
            plot_pie_chart(filterd_df_positive, labels)


if __name__ == "__main__":
    main()
