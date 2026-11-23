// Drizzle schema for this space.
//
// One table: `snapshots`. Each row is one generated Oura preventive health
// report, stored as a dated snapshot so past reports stay browsable. The
// report content itself (comparison table, band distribution, notes, chart
// series, recommendations) lives in `payload` as a JSON document whose shape
// is defined by `reportPayloadSchema` in ./report-schema.ts — the same shape
// the report generator produces and that `addSnapshot` accepts on refresh.

import { integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const snapshots = sqliteTable("snapshots", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  // Calendar dates of the Oura data window the report covers (YYYY-MM-DD).
  windowStart: text("window_start").notNull(),
  windowEnd: text("window_end").notNull(),
  // Human label for when this report was generated (YYYY-MM-DD).
  generatedLabel: text("generated_label").notNull(),
  // Full report payload as JSON (see ./report-schema.ts).
  payload: text("payload").notNull(),
  createdAt: integer("created_at", { mode: "timestamp_ms" })
    .notNull()
    .$defaultFn(() => new Date()),
});
