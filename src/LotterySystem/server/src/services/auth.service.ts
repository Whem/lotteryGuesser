import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';
import { prisma } from '../config/database';
import { config } from '../config/env';
import { AppError } from '../middleware/errorHandler';
import { JwtPayload } from '../middleware/auth';

export class AuthService {
  private static readonly SALT_ROUNDS = 12;

  static async register(email: string, password: string, displayName?: string) {
    // Check if user exists
    const existingUser = await prisma.user.findUnique({ where: { email } });
    if (existingUser) {
      throw new AppError('Email already registered', 400, 'EMAIL_EXISTS');
    }

    // Validate password
    if (password.length < 8) {
      throw new AppError('Password must be at least 8 characters', 400, 'WEAK_PASSWORD');
    }

    // Hash password
    const passwordHash = await bcrypt.hash(password, this.SALT_ROUNDS);

    // Create user with default settings
    const user = await prisma.user.create({
      data: {
        email,
        passwordHash,
        displayName: displayName || email.split('@')[0],
        settings: {
          create: {
            language: 'hu',
            darkMode: false,
            notificationsEnabled: true,
          },
        },
      },
      select: {
        id: true,
        email: true,
        displayName: true,
        isPremium: true,
        createdAt: true,
      },
    });

    // Generate tokens
    const tokens = await this.generateTokens(user.id, user.email);

    return { user, ...tokens };
  }

  static async login(email: string, password: string) {
    // Find user
    const user = await prisma.user.findUnique({
      where: { email },
      select: {
        id: true,
        email: true,
        passwordHash: true,
        displayName: true,
        isPremium: true,
        premiumExpiresAt: true,
        avatarUrl: true,
      },
    });

    if (!user) {
      throw new AppError('Invalid email or password', 401, 'INVALID_CREDENTIALS');
    }

    // Check password
    const isValidPassword = await bcrypt.compare(password, user.passwordHash);
    if (!isValidPassword) {
      throw new AppError('Invalid email or password', 401, 'INVALID_CREDENTIALS');
    }

    // Update last login
    await prisma.user.update({
      where: { id: user.id },
      data: { lastLoginAt: new Date() },
    });

    // Generate tokens
    const tokens = await this.generateTokens(user.id, user.email);

    // Remove passwordHash from response
    const { passwordHash: _, ...userWithoutPassword } = user;

    return { user: userWithoutPassword, ...tokens };
  }

  static async refreshTokens(refreshToken: string) {
    // Verify refresh token
    let decoded: JwtPayload;
    try {
      decoded = jwt.verify(refreshToken, config.jwtRefreshSecret) as JwtPayload;
    } catch {
      throw new AppError('Invalid refresh token', 401, 'INVALID_REFRESH_TOKEN');
    }

    // Check if token exists in database
    const storedToken = await prisma.refreshToken.findUnique({
      where: { token: refreshToken },
      include: { user: { select: { id: true, email: true } } },
    });

    if (!storedToken || storedToken.expiresAt < new Date()) {
      throw new AppError('Refresh token expired or invalid', 401, 'REFRESH_TOKEN_EXPIRED');
    }

    // Delete old refresh token
    await prisma.refreshToken.delete({ where: { id: storedToken.id } });

    // Generate new tokens
    return this.generateTokens(storedToken.user.id, storedToken.user.email);
  }

  static async logout(refreshToken: string) {
    await prisma.refreshToken.deleteMany({ where: { token: refreshToken } });
  }

  static async logoutAll(userId: string) {
    await prisma.refreshToken.deleteMany({ where: { userId } });
  }

  private static async generateTokens(userId: string, email: string) {
    const payload: JwtPayload = { userId, email };

    const accessToken = jwt.sign(payload, config.jwtSecret, {
      expiresIn: config.jwtExpiresIn as any,
    });

    const refreshToken = jwt.sign(payload, config.jwtRefreshSecret, {
      expiresIn: config.jwtRefreshExpiresIn as any,
    });

    // Store refresh token
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 7); // 7 days

    await prisma.refreshToken.create({
      data: {
        token: refreshToken,
        userId,
        expiresAt,
      },
    });

    return { accessToken, refreshToken };
  }

  static async changePassword(userId: string, currentPassword: string, newPassword: string) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { passwordHash: true },
    });

    if (!user) {
      throw new AppError('User not found', 404, 'USER_NOT_FOUND');
    }

    const isValidPassword = await bcrypt.compare(currentPassword, user.passwordHash);
    if (!isValidPassword) {
      throw new AppError('Current password is incorrect', 400, 'INVALID_PASSWORD');
    }

    if (newPassword.length < 8) {
      throw new AppError('New password must be at least 8 characters', 400, 'WEAK_PASSWORD');
    }

    const newPasswordHash = await bcrypt.hash(newPassword, this.SALT_ROUNDS);

    await prisma.user.update({
      where: { id: userId },
      data: { passwordHash: newPasswordHash },
    });

    // Invalidate all refresh tokens
    await this.logoutAll(userId);

    return { message: 'Password changed successfully' };
  }
}
