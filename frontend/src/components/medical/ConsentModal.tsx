import { useState, useEffect } from 'react';
import { Modal, Button } from '../ui';

interface ConsentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAccept: () => void;
}

export function ConsentModal({ isOpen, onClose, onAccept }: ConsentModalProps) {
  const [accepted, setAccepted] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('consentAccepted');
    if (stored) {
      setAccepted(true);
      onClose();
    }
  }, [onClose]);

  const handleAccept = () => {
    setAccepted(true);
    localStorage.setItem('consentAccepted', 'true');
    onAccept();
  };

  if (!isOpen || accepted) return null;

  return (
    <Modal
      isOpen={true}
      onClose={onClose}
      title="Informed Consent"
      description="Please review and accept before using CAAR-CDSS"
      hideCloseButton
      size="md"
    >
      <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
        <div className="bg-medical-neutral-50 dark:bg-medical-neutral-800 rounded-lg p-4">
          <h4 className="font-semibold text-medical-neutral-900 dark:text-medical-neutral-100 mb-2">
            About This System
          </h4>
          <p className="text-sm text-medical-neutral-600 dark:text-medical-neutral-400">
            CAAR-CDSS (Confidence-Aware Adaptive Retrieval Clinical Decision Support System) is a research
            prototype developed for academic evaluation of adaptive evidence retrieval in clinical decision support.
          </p>
        </div>

        <div className="space-y-3">
          <h4 className="font-semibold text-medical-neutral-900 dark:text-medical-neutral-100">
            Important Disclaimers
          </h4>
          <ul className="text-sm text-medical-neutral-600 dark:text-medical-neutral-400 space-y-2 list-disc list-inside">
            <li><strong>Not a medical device:</strong> This system is not approved by any regulatory authority for clinical use.</li>
            <li><strong>Research prototype:</strong> Results are for evaluation purposes only and should not guide patient care.</li>
            <li><strong>Clinician review required:</strong> All outputs must be verified by a qualified healthcare professional.</li>
            <li><strong>No liability:</strong> The developers assume no responsibility for clinical decisions made using this system.</li>
            <li><strong>Data privacy:</strong> No patient data is stored. Queries are processed in-memory for research evaluation.</li>
          </ul>
        </div>

        <div className="bg-medical-primary-50 dark:bg-medical-primary-900/30 border border-medical-primary-200 dark:border-medical-primary-800 rounded-lg p-4">
          <h4 className="font-semibold text-medical-primary-900 dark:text-medical-primary-100 mb-2">
            Your Agreement
          </h4>
          <p className="text-sm text-medical-primary-700 dark:text-medical-primary-300">
            By accepting, you acknowledge that you understand this is a research prototype and agree not to use
            its outputs for clinical decision making without independent verification by a qualified clinician.
          </p>
        </div>
      </div>

      <div className="mt-6 flex justify-end gap-3">
        <Button variant="ghost" onClick={onClose} disabled={!accepted}>
          Decline
        </Button>
        <Button variant="primary" onClick={handleAccept} disabled={!accepted}>
          Accept & Continue
        </Button>
      </div>

      <p className="mt-3 text-xs text-center text-medical-neutral-500 dark:text-medical-neutral-400">
        You can withdraw consent at any time by clearing your browser data.
      </p>
    </Modal>
  );
}