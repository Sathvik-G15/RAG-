import { test, expect } from '@playwright/test';

test.describe('Patient Portal', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[name="username"]', 'doctor');
    await page.fill('input[name="password"]', 'x');
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL('/');
  });

  test('should display patient portal form', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Patient Symptom Checker');
    await expect(page.locator('textarea[label="Symptoms"]')).toBeVisible();
    await expect(page.locator('input[label="Age"]')).toBeVisible();
    await expect(page.locator('select[label="Gender"]')).toBeVisible();
  });

  test('should analyze symptoms and show results', async ({ page }) => {
    await page.fill('textarea[label="Symptoms"]', '55-year-old diabetic male with central chest pain and sweating');
    await page.fill('input[label="Age"]', '55');
    await page.selectOption('select[label="Gender"]', 'male');
    await page.click('button:has-text("Analyze Symptoms")');

    await expect(page.locator('[role="status"]')).toBeVisible({ timeout: 30000 });

    await expect(page.locator('h2')).toBeVisible();
    await expect(page.locator('text=Confidence')).toBeVisible();
  });
});