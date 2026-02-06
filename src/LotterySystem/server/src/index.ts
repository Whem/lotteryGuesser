import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import path from 'path';
import { config } from './config/env';
import { logger } from './utils/logger';
import { errorHandler, notFoundHandler } from './middleware/errorHandler';
import { prisma } from './config/database';

// Routes
import authRoutes from './routes/auth.routes';
import userRoutes from './routes/user.routes';
import lotteryRoutes from './routes/lottery.routes';
import algorithmRoutes from './routes/algorithm.routes';
import generationRoutes from './routes/generation.routes';
import competitionRoutes from './routes/competition.routes';
import ticketRoutes from './routes/ticket.routes';
import adminRoutes from './routes/admin.routes';

const app = express();

// Security middleware - relaxed CSP for web frontend
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
      scriptSrcAttr: ["'unsafe-inline'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
      fontSrc: ["'self'", "https://fonts.gstatic.com", "data:"],
      imgSrc: ["'self'", "data:", "blob:", "http:", "https:"],
      connectSrc: ["'self'", "https:", "http:", "http://localhost:3000"],
      upgradeInsecureRequests: null, // DISABLE - causes SSL errors on HTTP
    },
  },
  crossOriginOpenerPolicy: false, // Disable COOP to avoid cluster warning
}));
app.use(cors({
  origin: config.corsOrigins,
  credentials: true,
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: config.rateLimitWindowMs,
  max: config.rateLimitMaxRequests,
  message: { error: 'Too many requests, please try again later.' },
});
app.use('/api/', limiter);

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Logging
app.use(morgan('combined', {
  stream: { write: (message) => logger.http(message.trim()) }
}));

// Static files (web frontend)
// In Docker: __dirname = /app/dist/src, so we go up 2 levels to /app/public
app.use(express.static(path.join(__dirname, '..', '..', 'public')));

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// API Routes
app.use('/api/auth', authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/lotteries', lotteryRoutes);
app.use('/api/algorithms', algorithmRoutes);
app.use('/api/generate', generationRoutes);
app.use('/api/competitions', competitionRoutes);
app.use('/api/tickets', ticketRoutes);
app.use('/api/admin', adminRoutes);

// Error handling
app.use(notFoundHandler);
app.use(errorHandler);

// Graceful shutdown
const shutdown = async () => {
  logger.info('Shutting down server...');
  await prisma.$disconnect();
  process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

// Start server
const start = async () => {
  try {
    await prisma.$connect();
    logger.info('Database connected successfully');
    
    app.listen(config.port, () => {
      logger.info(`🚀 Server running on port ${config.port} in ${config.nodeEnv} mode`);
      logger.info(`📊 Health check: http://localhost:${config.port}/health`);
    });
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
};

start();

export default app;
