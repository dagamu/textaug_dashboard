def row_elements(st, elements, common_params={}):
    result = [None]* len(elements)
    for i, item, col in zip( range(len(elements)), elements, st.columns(len(elements)) ):
        with col:
            result[i] = item["el"](**item["params"], **common_params)
    return result