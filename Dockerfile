FROM docker.arvancloud.ir/pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
LABEL authors="Navid Ziaei"
RUN addgroup app && adduser --system --ingroup app appuser
USER appuser

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /ride_conversion_predictor/
COPY . .
CMD ["python" ,"main.py"]
# ENTRYPOINT ["python" ,"main.py"]