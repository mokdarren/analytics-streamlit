# helper functions

import base64


# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, filename, message="Download CSV File",index=False):
    csv = df.to_csv(index=index)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{message}</a>'
    return href