"use client";

import { useRef } from "react";

type DropZoneProps = {
  file: File | null;
  onFileSelect: (file: File | null) => void;
};

export default function DropZone({ file, onFileSelect }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const openFilePicker = () => {
    if (inputRef.current) {
      // Reset input value so selecting the same file triggers onChange.
      inputRef.current.value = "";
      inputRef.current.click();
    }
  };

  const onDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      onFileSelect(dropped);
    }
  };

  return (
    <div
      className="drop-zone"
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
      onClick={openFilePicker}
      role="button"
      tabIndex={0}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept=".pdf,.docx,.jpg,.jpeg,.png"
        onChange={(e) => onFileSelect(e.target.files?.[0] || null)}
      />
      <p>
        {file
          ? `Selected: ${file.name}`
          : "Drag and drop a file here, or click to browse"}
      </p>
      <small>Accepted: PDF, DOCX, JPG, JPEG, PNG</small>
    </div>
  );
}
