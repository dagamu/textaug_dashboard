from steps.txt_processing_page import TEXT_ACTIONS, LABELS_ACTIONS
from steps.dataset_page import RenderPage as RenderLoadingDataset

def format_existing_dataset(session_state):
    return {
        "name": session_state["df_name"],
        "df": session_state["df"],
        "text_column": session_state["text_col"],
        "labels_column": session_state["labels_column"]
    } 

def render_dataset_info( st, dataset, del_dataset ):
    df_container = st.container()
    
    df = dataset['df']
    df_cols = df.columns
    
    with df_container:
        metadata_col, props_col, proc_col, del_col = st.columns([3,2,2,1])
        
        metadata_col.markdown(f"**{dataset['name']}**")
        metadata_col.markdown(f"N° Samples: {df.shape[0]}")
        
        prev_text_col_index = None
        if "text_column" in dataset:
            text_column = dataset["text_column"]
            prev_text_col_index = list(df_cols).index(text_column)
            
        prev_labels_col_index = None
        if "text_column" in dataset:
            labels_column = dataset["labels_column"]
            prev_labels_col_index = list(df_cols).index(labels_column)
            
        
        props_col.selectbox("Text Column", df_cols, index=prev_text_col_index)
        props_col.selectbox("Labels Column", df_cols, index=prev_labels_col_index)
        
        proc_col.selectbox("Apply to Text Column", [])
        proc_col.selectbox("Apply to Labels Column", [])
        
        if del_col.button("Delete", use_container_width=True ):
            del_dataset(dataset)
            return
            
        st.divider()

class DatasetList:
    datasets = []
    
    def __init__(self, st):
        self.st = st
        
        if "df" in st.session_state and "prev_datset_dropped" not in st.session_state:
            self.datasets.append( format_existing_dataset(st.session_state) )
    
    def add(self, df, df_name, text_col, labels_col):
        self.datasets.append({
            "df": df,
            "name": df_name,
            "text_column": text_col,
            "labels_column": labels_col
        })
        self.st.rerun()
        
    def delete(self, dataset):
        if dataset in self.datasets:
            if len(self.datasets) == 1:
                self.st.session_state["prev_datset_dropped"] = True
            self.datasets.remove(dataset)
        self.st.rerun()
        
    def render(self):
        st = self.st
        st.markdown("**Dataset Setting**")
        
        @st.dialog("Add Dataset")
        def add_dataset_dialog(st):
            RenderLoadingDataset(st, self.add)
            
        datasets_container = st.container(border=True)
        for dataset in self.datasets:
            render_dataset_info(datasets_container, dataset, self.delete )
            
        add_container = datasets_container.container(border=False)
        if add_container.button("Add Dataset", type="secondary"):
            add_dataset_dialog(st)
    
def PipelineFlow(st):
        
    st.subheader("Pipeline Flow")
    
    selected_datasets = DatasetList(st)
    selected_datasets.render()