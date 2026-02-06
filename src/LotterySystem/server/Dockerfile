# Single stage - dist már készen van
FROM node:20-alpine

# OpenSSL telepítése a Prisma-hoz
RUN apk add --no-cache openssl openssl-dev

WORKDIR /app

ENV NODE_ENV=production

# Copy package files
COPY package*.json ./

# Copy pre-built dist, prisma, and public (web frontend)
COPY dist ./dist
COPY prisma ./prisma
COPY public ./public

# Install production dependencies
RUN npm ci --only=production

# Generate Prisma client
RUN npx prisma generate

# Jogosultságok beállítása
RUN chown -R node:node /app

# Node user használata
USER node

EXPOSE 3000

CMD ["node", "dist/src/index.js"]
