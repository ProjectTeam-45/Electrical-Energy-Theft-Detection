FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install streamlit pandas numpy scikit-learn==1.3.2 xgboost joblib plotly

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
