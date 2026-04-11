/**
 * Mail Service
 * 
 * Handles sending emails for password recovery and notifications.
 */

export class MailService {
  /**
   * Sends a password reset email
   */
  static async sendPasswordResetEmail(email, token) {
    const resetLink = `http://localhost:3000/password/reset?token=${token}`;
    
    console.log('-----------------------------------------');
    console.log(`To: ${email}`);
    console.log('Subject: NEPSE Analysis - Password Reset');
    console.log(`Content: Use this link to reset your password: ${resetLink}`);
    console.log('Note: This link will expire in 1 hour.');
    console.log('-----------------------------------------');
    
    // In production, use nodemailer to send a real email
    return true;
  }

  /**
   * Sends an MFA setup notification
   */
  static async sendMfaEnabledNotification(email) {
    console.log(`MFA has been enabled for ${email}`);
    return true;
  }
}
