FROM python:3.11

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/orders-api

ARG PIP_EXTRA_INDEX_URL

# Install requirements, including from Gemfury
ADD ./orders-api /opt/orders-api/
RUN pip install --upgrade pip
RUN pip install -r /opt/orders-api/requirements/requirements.txt

RUN chmod +x /opt/orders-api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]