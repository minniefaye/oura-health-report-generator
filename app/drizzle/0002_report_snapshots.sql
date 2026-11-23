DROP TABLE IF EXISTS `entries`;
--> statement-breakpoint
CREATE TABLE `snapshots` (
  `id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
  `window_start` text NOT NULL,
  `window_end` text NOT NULL,
  `generated_label` text NOT NULL,
  `payload` text NOT NULL,
  `created_at` integer NOT NULL
);
