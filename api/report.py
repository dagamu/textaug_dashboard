import pandas as pd
import re

COL_FORMAT_DICT = {
    "dataset": "Dataset",
    "vec_method": "Representation Method",
    "pt_method": "Problem Trasformation Method",
    "model": "Model",
    "aug_method": "Augmentation Method",
    "aug_choice_method": "Augmentation Input Choice Method" 
}

METRICS_FORMAT_DICT = {
    "acc": "Accuracy",
    "exact_acc": "Exact Accuracy",
    "hl": "Hamming Loss",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score"
}

class ReportGenerator:
    
    group_by = None
    lines = None
    
    def __init__(self, session):
        self.session = session
        self.loaded = False
        self.data = {}
        
    def discretize(self, column, n, label=None):
        data = pd.DataFrame(self.data)
        label = column if label == None else label
        data[f"{column}_groups"] = pd.qcut(data[column], n, precision=2) \
                                            .map(lambda v: f"{round(v.left, 2)}<={label}<{round(v.right, 2)}")
        self.data = data.to_dict("records")
        
    def set_results(self, data):
        self.data = data
        self.discretize("base_mean_ir", 4, "MeanIR")
        self.loaded = True
        
    def clear_results(self):
        self.data = {}
        self.loaded = False
        
    def render(self, viewer):
        
        results = self.data
        
        viewer.title("Pipeline Results")
        report_tab, table_tab = viewer.sections(["Report", "Table"])
        
        results_df = pd.DataFrame(results)
        table_tab.dataframe(results_df)
        
        with report_tab:
            graph_col, setting_col = viewer.columns([5,2])
            
            with setting_col:
                num_cols = results_df._get_numeric_data().columns
                categorical_cols = set(results_df.columns) - set(num_cols)
                
                group_by_options = set([None] + [col for col in categorical_cols if len(results_df[col].unique()) > 1 ])
                self.group_by = viewer.selectbox("Group by", group_by_options - set([self.lines]), format_func=lambda x: COL_FORMAT_DICT.get(x) or x )
                self.lines = viewer.selectbox("Lines", group_by_options - set([self.group_by]), format_func=lambda x: COL_FORMAT_DICT.get(x) or x)
                performance_metric = viewer.selectbox("Performance Metric", ["acc", "exact_acc", "hl", "precision", "recall", "f1"], format_func=lambda x: METRICS_FORMAT_DICT.get(x) or x)
                
            with graph_col:
                
                pattern = rf"^[^_]+_{performance_metric}$"
                metric_cols = [col for col in results_df.columns if re.search(pattern, col) ]
                format_cols = [ f"*{col.split('_')[0]}" for col in metric_cols ]
                
                if self.group_by == None:
                    y_values = results_df[metric_cols].mean()
                    fig = viewer.line_chart(format_cols, y_values)
                
                else:
                    fig = viewer.subplots(results_df, self.group_by, self.lines, metric_cols, format_cols)
                    
                viewer.set_plots(fig, yaxis_title=METRICS_FORMAT_DICT[performance_metric])
                
                