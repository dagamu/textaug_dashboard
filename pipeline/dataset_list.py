import streamlit as st

from pages.dataset_loading import RenderPage as RenderLoadingDataset
from pages.text_processing import RenderPage as RenderTxtProcSelection

class DatasetList:
    datasets = []
    
    def __init__(self, st):
        self.st = st
        
        if "df" in st.session_state and "prev_dataset_dropped" not in st.session_state:
            df_session_keys = ["df", "df_name", "text_col", "labels_column"]
            df_session = [ st.session_state[item] for item in df_session_keys ]
            self.add(*df_session) 
    
    def add(self, df, df_name, text_col, labels_col):
        if df_name in [ item["name"] for item in self.datasets ]:
            return 
        
        self.datasets.append({
            "df": df,
            "name": df_name,
            "text_column": text_col,
            "labels_column": labels_col,
            "text_proc_action": None,
            "labels_proc_action": None,
        })
        
        st.session_state["pipeline_datasets"] = self
        st.rerun()
        
    def delete(self, dataset):
        self.datasets = [ item for item in self.datasets if item["name"] != dataset["name"] ]
        
        if "df_name" in st.session_state:
            if dataset["name"] == st.session_state["df_name"]:
                self.st.session_state["prev_dataset_dropped"] = True
        
        st.session_state["pipeline_datasets"] = self 
        st.rerun()
        
        
    def set_action(self, df_name, fn, col):
        for dataset in self.datasets:
            if dataset["name"] == df_name:
                if col == dataset["text_column"]:
                    dataset["text_proc_action"] = fn
                elif col == dataset["labels_column"]:
                    dataset["labels_proc_action"] = fn
        st.toast(f"{df_name, fn.name, col}")
        st.session_state["pipeline_datasets"] = self 
        st.rerun()
                    
    @st.dialog("Add Dataset")
    def add_dataset_dialog(self):
        RenderLoadingDataset(self.st, self.add)
        
    @st.dialog("Set Text Preprocessing Action")
    def proc_action_diag(self, df_name, col):
        set_fn = lambda fn, col: self.set_action(df_name, fn, col)
        RenderTxtProcSelection(self.st, [col], set_fn )
        
    def render_item( self, stc, dataset ):
        
        df = dataset['df']
        df_cols = df.columns
        
        item_container = stc.container()
        
        metadata_col, props_col, proc_col, del_col = item_container.columns([3,2,2,1])
        
        df_name = dataset['name']
        metadata_col.markdown(f"**{df_name}**")
        metadata_col.markdown(f"N° Samples: {df.shape[0]}")
        
        prev_text_col_index = None
        if "text_column" in dataset:
            text_column = dataset["text_column"]
            prev_text_col_index = list(df_cols).index(text_column)
            
        prev_labels_col_index = None
        if "text_column" in dataset:
            labels_column = dataset["labels_column"]
            prev_labels_col_index = list(df_cols).index(labels_column)
            
        selected_text_col = props_col.selectbox("Text Column", df_cols, index=prev_text_col_index, key=f"TC-{dataset['name']}")
        selected_labels_col = props_col.selectbox("Labels Column", df_cols, index=prev_labels_col_index, key=f"LC-{dataset['name']}")
        
        
        txt_action_name = "None"
        if dataset["text_proc_action"] != None:
            txt_action_name = dataset["text_proc_action"].name
        
        labels_action_name = "None"
        if dataset["labels_proc_action"] != None:
            labels_action_name = dataset["labels_proc_action"].name

        
        if proc_col.button(f"{txt_action_name} | ✏", key=f"TCF-{dataset['name']}"):
            self.proc_action_diag(dataset['name'], selected_text_col)
            
        proc_col.caption("_")
        
        if proc_col.button(f"{labels_action_name} | ✏", key=f"LCF-{dataset['name']}"):
            self.proc_action_diag(dataset['name'], selected_labels_col)
        
        del_col.text("")
        if del_col.button("🗑", use_container_width=True, key=f"DELBTN-{dataset['name']}"):
            self.delete(dataset)
            return
            
        stc.divider()  
        
    def render(self):
        stc = st.container()
        
        with stc:
            st.markdown("**Dataset Setting**")
            #st.write(len(self.datasets))
                
            datasets_container = st.container(border=True)
            for dataset in self.datasets:
                self.render_item(datasets_container.container(), dataset )
                
            if datasets_container.button("Add Dataset", type="secondary"):
                self.add_dataset_dialog()