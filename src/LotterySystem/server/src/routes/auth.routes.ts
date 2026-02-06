import { Router, Request, Response } from 'express';
import { body, validationResult } from 'express-validator';
import { asyncHandler } from '../middleware/errorHandler';
import { authenticate, AuthRequest } from '../middleware/auth';
import { AuthService } from '../services/auth.service';

const router = Router();

// Validation middleware
const validateRequest = (req: Request, res: Response, next: Function) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Invalid input',
        details: errors.array(),
      },
    });
  }
  next();
};

// Register
router.post(
  '/register',
  [
    body('email').isEmail().normalizeEmail(),
    body('password').isLength({ min: 8 }),
    body('displayName').optional().isLength({ min: 2, max: 50 }),
  ],
  validateRequest,
  asyncHandler(async (req: Request, res: Response) => {
    const { email, password, displayName } = req.body;
    const result = await AuthService.register(email, password, displayName);
    res.status(201).json({ success: true, data: result });
  })
);

// Login
router.post(
  '/login',
  [
    body('email').isEmail().normalizeEmail(),
    body('password').notEmpty(),
  ],
  validateRequest,
  asyncHandler(async (req: Request, res: Response) => {
    const { email, password } = req.body;
    const result = await AuthService.login(email, password);
    res.json({ success: true, data: result });
  })
);

// Refresh token
router.post(
  '/refresh',
  [body('refreshToken').notEmpty()],
  validateRequest,
  asyncHandler(async (req: Request, res: Response) => {
    const { refreshToken } = req.body;
    const result = await AuthService.refreshTokens(refreshToken);
    res.json({ success: true, data: result });
  })
);

// Logout
router.post(
  '/logout',
  [body('refreshToken').notEmpty()],
  validateRequest,
  asyncHandler(async (req: Request, res: Response) => {
    const { refreshToken } = req.body;
    await AuthService.logout(refreshToken);
    res.json({ success: true, message: 'Logged out successfully' });
  })
);

// Logout all sessions
router.post(
  '/logout-all',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    await AuthService.logoutAll(req.user!.id);
    res.json({ success: true, message: 'Logged out from all devices' });
  })
);

// Change password
router.post(
  '/change-password',
  authenticate,
  [
    body('currentPassword').notEmpty(),
    body('newPassword').isLength({ min: 8 }),
  ],
  validateRequest,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { currentPassword, newPassword } = req.body;
    const result = await AuthService.changePassword(req.user!.id, currentPassword, newPassword);
    res.json({ success: true, data: result });
  })
);

// Get current user
router.get(
  '/me',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    res.json({ success: true, data: req.user });
  })
);

export default router;
