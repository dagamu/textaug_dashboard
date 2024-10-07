LOADING_PRESETS = {
    "data_folder":
        {
            "Code Tutorials Multilabel.csv": {
                "cols_to_drop": ["mysql", "python", "php"]
            }
        }
}

def row_elements(st, elements, common_params={}):
    result = [None]* len(elements)
    for i, item, col in zip( range(len(elements)), elements, st.columns(len(elements)) ):
        with col:
            result[i] = item["el"](**item["params"], **common_params)
    return result