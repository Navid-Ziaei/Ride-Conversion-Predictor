FROM docker.arvancloud.ir/pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

LABEL authors="Navid Ziaei"
RUN addgroup app && adduser --system --ingroup app appuser
USER appuser

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /ride_conversion_predictor/
COPY --chown=appuser:app . .
COPY --chown=appuser:app configs .

# Create the device_path.yaml file with the required paths
RUN mkdir -p /ride_conversion_predictor/configs && \
    echo "raw_dataset_path: \"/ride_conversion_predictor/data/DSM/\"" > /ride_conversion_predictor/configs/device_path.yaml && \
    echo "preprocessed_dataset_path: \"/ride_conversion_predictor/data/DSM_preprocessed/\"" >> /ride_conversion_predictor/configs/device_path.yaml && \
    echo "model_path: \"/ride_conversion_predictor/saved_model/\"" >> /ride_conversion_predictor/configs/device_path.yaml

# Declare volumes
VOLUME /ride_conversion_predictor/data
VOLUME /ride_conversion_predictor/results

# CMD ["python" ,"main.py"]
ENTRYPOINT ["python" ,"main.py"]