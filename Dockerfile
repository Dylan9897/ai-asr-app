FROM asr:v1.0
ADD . /workspace
WORKDIR /workspace
ENV PATH=/root/miniconda3/bin:$PATH
EXPOSE 18002
RUN pip install aiohttp
RUN chmod +x /workspace/run.sh
ENTRYPOINT ["bash","run.sh"]