import type { WizardStep } from "../lib/types";

const steps: { id: WizardStep; label: string; eyebrow: string }[] = [
  { id: "upload", label: "Upload", eyebrow: "01" },
  { id: "mapping", label: "Mapping", eyebrow: "02" },
  { id: "processing", label: "Processing", eyebrow: "03" },
  { id: "results", label: "Results", eyebrow: "04" }
];

export function StepRail({ currentStep }: { currentStep: WizardStep }) {
  const currentIndex = steps.findIndex((step) => step.id === currentStep);

  return (
    <div className="step-rail glass-panel">
      {steps.map((step, index) => {
        const isActive = step.id === currentStep;
        const isComplete = index < currentIndex;

        return (
          <div
            key={step.id}
            className={[
              "step-node",
              isActive ? "step-node-active" : "",
              isComplete ? "step-node-complete" : ""
            ].join(" ")}
          >
            <span className="step-node-eyebrow">{step.eyebrow}</span>
            <span className="step-node-label">{step.label}</span>
          </div>
        );
      })}
    </div>
  );
}
