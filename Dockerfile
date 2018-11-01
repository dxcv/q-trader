FROM python:3.6.7-slim-stretch

RUN apt-get update && apt-get -y install curl build-essential && apt-get clean
# Install TA-lib
RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | \
  tar xzvf - && \
  cd ta-lib && \
  sed -i "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h && \
  ./configure && make && make install && \
  cd .. && rm -rf ta-lib
ENV LD_LIBRARY_PATH /usr/local/lib

# Prepare environment
RUN mkdir /q-trader
# Set the working directory
WORKDIR /q-trader

# Install any needed packages specified in requirements.txt
COPY requirements.txt /q-trader
RUN pip install -r requirements.txt --no-cache-dir

# Copy the current directory contents into the container at /app
COPY . /q-trader

# Run app.py when the container launches
CMD ["python", "bot.py"]
