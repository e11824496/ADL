import streamlit as st
import pandas as pd
import json
import requests

API_URL = "http://localhost:8000/classify"


def get_prediction():
    predictions = st.session_state.predictions
    current_line = st.session_state.current_line
    return predictions.get(current_line) if predictions else None


def fetch_prediction_data(line):
    row = st.session_state.data.iloc[line]
    return {
        'amount': row.get('Amount', 0),
        'description': row.get('Description', ''),
        'time': row.get('Time', '')
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


def fetch_next_predictions():
    max_lines = 10
    current_line = st.session_state.current_line
    data_index_len = len(st.session_state.data.index)

    for i in range(max_lines + 1):
        line = (current_line + i) % data_index_len

        if line in st.session_state.labels.index or \
                line in st.session_state.predictions:
            continue

        data = fetch_prediction_data(line)
        predictions = make_prediction_request(data)

        if predictions:
            st.session_state.predictions[current_line + i] = predictions
            if i == 0:
                st.rerun()


@st.cache_data
def load_categories(file_path) -> dict:
    with open(file_path) as f:
        categories = json.load(f)
    return categories


@st.cache_data
def load_data(file_path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data


def select_group(category):
    st.session_state.category_group = category
    st.session_state.category_subgroup = None


def select_subgroup(category):
    st.session_state.category_subgroup = category


def set_data():
    data_file = st.session_state.dataset_file
    st.session_state.data = load_data(data_file)
    set_labels()


def set_labels():
    data_file = st.session_state.labels_file
    st.session_state.labels = pd.DataFrame({
        'category_group': [],
        'category_subgroup': []
    })
    if data_file is not None:
        labels = load_data(data_file)
        # check if columns contain 'category_group' and 'category_subgroup'
        if 'category_group' in labels.columns and \
                'category_subgroup' in labels.columns:
            st.session_state.labels = labels
        else:
            st.warning("Labels file needs columns 'category_group' and " +
                       "'category_subgroup'")
    st.session_state.current_line = len(st.session_state.labels.index)


def submit():
    labels = st.session_state.labels
    cl = st.session_state.current_line
    labels.loc[cl] = {
        'category_group': st.session_state.category_group,
        'category_subgroup': st.session_state.category_subgroup
    }
    st.session_state.predictions.pop(cl, None)
    st.session_state.category_group = None
    st.session_state.category_subgroup = None
    st.session_state.current_line += 1


def change_line(diff):
    st.session_state.current_line += diff
    st.session_state.current_line = \
        st.session_state.current_line % len(st.session_state.data.index)

    st.session_state.category_group = None
    st.session_state.category_subgroup = None

    if st.session_state.current_line in st.session_state.labels.index:
        st.session_state.category_group = \
            st.session_state.labels.loc[st.session_state.current_line,
                                        'category_group']
        st.session_state.category_subgroup = \
            st.session_state.labels.loc[st.session_state.current_line,
                                        'category_subgroup']


def main():

    st.session_state.setdefault("category_group", None)
    st.session_state.setdefault("category_subgroup", None)
    st.session_state.setdefault("current_line", 0)
    st.session_state.setdefault("data", None)
    st.session_state.setdefault("labels", pd.DataFrame({
        'category_group': [],
        'category_subgroup': []
    }))
    st.session_state.setdefault("predictions", {})

    CATEGORIES = load_categories("../categories.json")

    st.title("Bank Statement Labeling Tool")

    st.sidebar.file_uploader(
        "Upload your dataset", type=["csv"], key="dataset_file",
        on_change=set_data)

    st.sidebar.file_uploader(
        "Upload your existing labels", type=["csv"], key="labels_file",
        on_change=set_labels)

    st.sidebar.download_button(
        "Download labels", data=st.session_state.labels.to_csv(),
        file_name="labels.csv", mime="text/csv")

    if st.session_state.data is not None:
        data = st.session_state.data

        prediction = get_prediction()
        if prediction and st.session_state.category_group is None:
            st.session_state.category_group = prediction[0]
            st.session_state.category_subgroup = prediction[1]

        st.subheader("Data")

        n_cols = len(CATEGORIES)//2
        cols = st.columns(n_cols)
        cols[0].button("Previous", on_click=change_line, args=(-1,),
                       use_container_width=True)
        cols[-1].button("Next", on_click=change_line, args=(1,),
                        use_container_width=True)

        row = data.iloc[st.session_state.current_line]
        row = row[['Time', 'Description', 'Amount']]
        row = row.apply(lambda x: str(x))
        st.dataframe(row,
                     use_container_width=True)

        st.subheader("Category Selection")
        cols = st.columns(n_cols)

        for i, group in enumerate(CATEGORIES):
            highlight = group == st.session_state.category_group
            cols[i % n_cols].button(
                group, on_click=select_group, args=(group,),
                use_container_width=True,
                type='primary' if highlight else 'secondary')

        if st.session_state.category_group:
            st.write("\n")
            st.markdown(
                f"**Subcategories of {st.session_state.category_group}:**")
            sub_categories = CATEGORIES[st.session_state.category_group]

            cols = st.columns(n_cols)
            for i, sub_group in enumerate(sub_categories):
                highlight = sub_group == st.session_state.category_subgroup
                cols[i % n_cols].button(
                    sub_group, key='sub-'+sub_group,
                    on_click=select_subgroup, args=(sub_group,),
                    use_container_width=True,
                    type='primary' if highlight else 'secondary')

        if st.session_state.category_subgroup:
            text = 'Edit' if st.session_state.current_line in \
                st.session_state.labels.index else 'Submit'
            st.button(text,
                      use_container_width=True, type='primary',
                      on_click=submit)

        fetch_next_predictions()


if __name__ == "__main__":
    main()
