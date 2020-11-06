FROM ubuntu:16.04


RUN apt-get update && apt-get install -y \
		    build-essential \
		    make \
            cmake \ 
            git \ 
            libeigen3-dev \ 
            vim       


WORKDIR /usr/src

RUN git clone https://github.com/opencv/opencv.git
RUN mkdir opencv/build
WORKDIR /usr/src/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=Release -D ENABLE_CXX11=ON ../ 
RUN make
RUN make install

ENV LocalExp /usr/src/LocalExp
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:${LocalExp}
RUN mkdir ${LocalExp}      

ADD ./ ${LocalExp}/Code
RUN mkdir -p ${LocalExp}/Code/build
WORKDIR ${LocalExp}/Code/build/
RUN cmake ../ 
RUN make

WORKDIR ${LocalExp}

RUN cp ${LocalExp}/Code/build/LocalExp* ${LocalExp}
