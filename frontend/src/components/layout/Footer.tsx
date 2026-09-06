export function Footer() {
  return (
    <footer className="border-t border-medical-neutral-200 dark:border-medical-neutral-700 bg-white dark:bg-medical-neutral-900">
      <div className="container-page py-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-sm text-medical-neutral-500 dark:text-medical-neutral-400">
            CAAR-CDSS v0.1.0 &mdash; Research prototype for academic use.
          </p>
          <p className="text-sm text-medical-neutral-500 dark:text-medical-neutral-400">
            Not a medical device. Do not use for clinical decision making.
          </p>
          <div className="flex items-center gap-4 text-sm">
            <a
              href="#"
              className="text-medical-neutral-500 hover:text-medical-primary-600 dark:text-medical-neutral-400 dark:hover:text-medical-primary-400"
            >
              Privacy
            </a>
            <a
              href="#"
              className="text-medical-neutral-500 hover:text-medical-primary-600 dark:text-medical-neutral-400 dark:hover:text-medical-primary-400"
            >
              Terms
            </a>
            <a
              href="#"
              className="text-medical-neutral-500 hover:text-medical-primary-600 dark:text-medical-neutral-400 dark:hover:text-medical-primary-400"
            >
              Contact
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}