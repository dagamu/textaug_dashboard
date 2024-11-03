import streamlit as st

from pages.data_augmentation import RenderPage as RenderAugSelection

def AugInputSelection():
    st.markdown("**Augmentation Input Selection**")
    
    select_method = st.multiselect("Input Selection", ["Random", "TF-IDF", "TF-muRFL"]) # TODO: +Perfomance Metric

    return select_method

def AugStepsBar():
    st.markdown("**Augmentation Steps**")
    
    kind_col, steps_col, custom_val_col, custom_btn_col = st.columns([2,3,1,1])
        
    def set_avaibable_steps( update=False ):
        st.session_state["pipeline_aug_steps"] = st.session_state["selected_aug_steps"] if update else []
        
    if not "pipeline_aug_steps" in st.session_state.keys():
         set_avaibable_steps()
         
    kind_labels = {"count": "Count (N)", "ratio": "Ratio (%)"}
    kind = kind_col.selectbox("Augmentation Kind", ["count", "ratio"], format_func=kind_labels.get, on_change=set_avaibable_steps)
    
    aviable_steps = st.session_state["pipeline_aug_steps"]
    
    suffix = '%' if kind == 'ratio' else ''
    format_steps = lambda val: f"+{val}{suffix}"
    steps = steps_col.multiselect("Steps", aviable_steps, aviable_steps, format_func=format_steps, on_change=lambda: set_avaibable_steps(True), key="selected_aug_steps" )
    
    val_args = { 
            "count": { "label": "Add Step (+N)", "value":   50,  "min_value": 1    , "step": 10, },
            "ratio": { "label": "Add Step (+%)", "value": 20.0, "min_value": 1.1, "step": 10.0 }
        }    
    custom_val = custom_val_col.number_input( **val_args[kind] )
    
    custom_btn_col.container(height=12, border=False)
    if custom_btn_col.button("Add", use_container_width=True):
        if not custom_val in aviable_steps:
            aviable_steps.append(custom_val)
            st.rerun()
    
    return kind, steps

class DataAugmentationList:
    methods = []
    
    def add(self, method):
        self.methods.append({
            "method": method,
            "label": method.name
        })
        st.session_state["pipeline_aug_methods"] = self
        st.rerun()
        
    def setup_aug_bar(self, method, _):
        if st.button("Initialize", key=f"PipelineAug-{method.name}"):
            self.add(method)
        
    def delete(self, method ):
        self.methods = [ item for item in self.methods if item["label"] != method["label"] ]
        st.session_state["pipeline_aug_methods"] = self 
        st.rerun()
        
    @st.dialog("Add Data Augmentation Method", width="large")
    def add_method_dialog(self):
        RenderAugSelection(st, self.setup_aug_bar, False)
        
    def render_item( self, stc, method ):
        item_container = stc.container()
        
        metadata_col, del_col = item_container.columns([4,1])
        
        metadata_col.markdown(f"**{method['label']}**")
        
        if del_col.button("🗑", use_container_width=True, key=f"DELBTN-{method['label']}"):
            self.delete(method)
            return
            
        stc.divider()  
        
    def render(self):
        stc = st.container()
        
        with stc:
            st.markdown("**Data Augmentation Methods**")
                
            methods_container = st.container(border=True)
            for method in self.methods:
                self.render_item( methods_container.container(), method )
                
            if methods_container.button("Add Method", type="secondary"):
                self.add_method_dialog()
                    
            aug_settings = st.container(border=True)
            with aug_settings:
                self.select_input_method = AugInputSelection()
                self.aug_kind, self.steps = AugStepsBar()