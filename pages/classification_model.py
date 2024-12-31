import streamlit as st
from menu import menu

def render_method(clf_manager, method, col_weights=[1,2]):
    del_col, label_col = st.columns(col_weights)
    del_col.button("🗑", use_container_width=True, key=f"delbtn_{method.key}", on_click=lambda: clf_manager.del_method(method) )
    label_col.button(f"⚙ {method.name}", key=f"config_{method.key}", use_container_width=False)
    
def selected_models():
    clf_manager = st.session_state["session"].classification
    repr_col, pt_col = st.columns(2)
    with repr_col.container(border=True):
        st.markdown("**1. Text Representation**")
        
        if len(clf_manager.selected_methods["text_representation"]) == 0:
            st.text("None")
            
        for method in clf_manager.selected_methods["text_representation"]:
            render_method(clf_manager, method)
            
    with pt_col.container(border=True):
        st.markdown("**3. Problem Transformation**")
        
        if len(clf_manager.selected_methods["problem_transformation"]) == 0:
            st.text("None")
            
        for method in clf_manager.selected_methods["problem_transformation"]:
            render_method(clf_manager, method)
            
    with st.container(border=True):
        st.markdown("**2. Classification Model**")
        
        if len(clf_manager.selected_methods["classification_model"]) == 0:
            st.text("None")
        
        for method in clf_manager.selected_methods["classification_model"]:
            render_method(clf_manager, method, col_weights=[1,5])

def ClasificationModelPage():
            
    st.title("Classification Model")
    clf_manager = st.session_state["session"].classification
    
    available_methods = clf_manager.available_methods
    
    selection_col, btn_col = st.columns([3,1])
    tr_selected = selection_col.selectbox("Text Representation", available_methods["text_representation"].keys()) 
    btn_col.container(border=False, height=10)
    btn_col.button("Add", key=f"tr_btn", on_click=lambda: clf_manager.add_method("text_representation", tr_selected) )
    
    selection_col, btn_col = st.columns([3,1])
    model_selected = selection_col.selectbox("Select Model to Train", available_methods["classification_model"].keys()) 
    btn_col.container(border=False, height=10)
    btn_col.button("Add", key=f"model_btn", on_click=lambda: clf_manager.add_method("classification_model", model_selected) )
    
    selection_col, btn_col = st.columns([3,1])
    pt_selected = selection_col.selectbox("Problem Transformation Method", available_methods["problem_transformation"].keys()) 
    btn_col.container(border=False, height=10)
    btn_col.button("Add", key=f"pt_btn", on_click=lambda: clf_manager.add_method("problem_transformation", pt_selected) )
   
    st.divider()
    selected_models()    
    
if __name__ == "__main__":         
    menu()
    ClasificationModelPage()