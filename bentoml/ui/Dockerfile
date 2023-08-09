FROM node:12.22.9-alpine

WORKDIR /ui

COPY package.json .

RUN npm install

COPY server.js .

CMD [ "node", "server.js" ]

EXPOSE 4001