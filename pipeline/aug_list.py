import streamlit as st

from pages.data_augmentation import RenderPage as RenderAugSelection


class DataAugmentationList:
    methods = []
    
    def add(self, method):
        self.methods.append({
            "method": method,
            "label": method.name
        })
        st.session_state["pipeline_aug_methods"] = self
        st.rerun()
        
    def delete(self, method ):
        self.methods = [ item for item in self.methods if item["label"] != method["label"] ]
        st.session_state["pipeline_aug_methods"] = self 
        st.rerun()
        
    @st.dialog("Add Data Augmentation Method", width="large")
    def add_method_dialog(self):
        RenderAugSelection(st, self.add, False)
        
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