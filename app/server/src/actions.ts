// Actions for this artifact. One file, one exported `Actions` map.
//
// The app stores Oura preventive health reports as dated snapshots. It
// never calls Oura itself: Muse refreshes it by pulling fresh data
// through the Oura connector, running the report generator, and calling
// `addSnapshot` with the new report payload — the same shape the seed
// snapshot uses (see ./report-schema.ts and ./seed-data.ts).

import { defineAction, z, type ActionsModule } from "@hatch/space-sdk";
import { desc, eq } from "drizzle-orm";
import * as schema from "./schema";
import { reportPayloadSchema, type ReportPayload } from "./report-schema";
import { SEED_SNAPSHOT } from "./seed-data";

const snapshotMeta = z.object({
  id: z.number(),
  windowStart: z.string(),
  windowEnd: z.string(),
  generatedLabel: z.string(),
});

const snapshotFull = snapshotMeta.extend({ payload: reportPayloadSchema });

function parsePayload(raw: string): ReportPayload {
  return reportPayloadSchema.parse(JSON.parse(raw) as unknown);
}

export const Actions = {
  listSnapshots: defineAction({
    description:
      "Lists all stored report snapshots (data window + generation date), newest data window first.",
    request: z.object({}),
    response: z.object({ snapshots: z.array(snapshotMeta) }),
    async handler(ctx) {
      const db = ctx.db<typeof schema>();
      const rows = await db
        .select({
          id: schema.snapshots.id,
          windowStart: schema.snapshots.windowStart,
          windowEnd: schema.snapshots.windowEnd,
          generatedLabel: schema.snapshots.generatedLabel,
        })
        .from(schema.snapshots)
        .orderBy(desc(schema.snapshots.windowEnd), desc(schema.snapshots.id));
      return { snapshots: rows };
    },
  }),

  getSnapshot: defineAction({
    description:
      "Returns one report snapshot with its full payload (comparison table, bands, notes, chart series, recommendations).",
    request: z.object({ id: z.number().int().positive() }),
    response: z.object({ snapshot: snapshotFull.nullable() }),
    async handler(ctx, args) {
      const db = ctx.db<typeof schema>();
      const rows = await db
        .select()
        .from(schema.snapshots)
        .where(eq(schema.snapshots.id, args.id))
        .limit(1);
      const row = rows[0];
      if (!row) return { snapshot: null };
      return {
        snapshot: {
          id: row.id,
          windowStart: row.windowStart,
          windowEnd: row.windowEnd,
          generatedLabel: row.generatedLabel,
          payload: parsePayload(row.payload),
        },
      };
    },
  }),

  addSnapshot: defineAction({
    description:
      "Adds a new report snapshot from a generated report payload. If a snapshot for the same data window already exists, it is replaced. This is the refresh path Muse uses after generating a fresh report.",
    request: z.object({
      windowStart: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
      windowEnd: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
      generatedLabel: z.string().min(1).max(40),
      payload: reportPayloadSchema,
    }),
    response: z.object({ id: z.number() }),
    async handler(ctx, args) {
      const db = ctx.db<typeof schema>();
      const payloadJson = JSON.stringify(args.payload);
      // Dedupe on the exact data window: a re-generated report for the
      // same window replaces the old snapshot instead of duplicating it.
      const all = await db
        .select({
          id: schema.snapshots.id,
          windowStart: schema.snapshots.windowStart,
          windowEnd: schema.snapshots.windowEnd,
        })
        .from(schema.snapshots);
      const dupe = all.find(
        (r) => r.windowStart === args.windowStart && r.windowEnd === args.windowEnd,
      );
      if (dupe) {
        await db
          .update(schema.snapshots)
          .set({
            generatedLabel: args.generatedLabel,
            payload: payloadJson,
          })
          .where(eq(schema.snapshots.id, dupe.id));
        ctx.invalidateQueries();
        return { id: dupe.id };
      }
      const result = await db
        .insert(schema.snapshots)
        .values({
          windowStart: args.windowStart,
          windowEnd: args.windowEnd,
          generatedLabel: args.generatedLabel,
          payload: payloadJson,
        })
        .returning({ id: schema.snapshots.id });
      const inserted = result[0];
      if (!inserted) throw new Error("addSnapshot: insert returned no rows");
      ctx.invalidateQueries();
      return { id: inserted.id };
    },
  }),

  seedSnapshot: defineAction({
    description:
      "Seeds the bundled first report snapshot, but only when no snapshots exist yet. Safe to call repeatedly; never overwrites stored reports.",
    request: z.object({}),
    response: z.object({ seeded: z.boolean(), id: z.number().nullable() }),
    async handler(ctx) {
      const db = ctx.db<typeof schema>();
      const existing = await db
        .select({ id: schema.snapshots.id })
        .from(schema.snapshots)
        .limit(1);
      if (existing.length > 0) return { seeded: false, id: null };
      const result = await db
        .insert(schema.snapshots)
        .values({
          windowStart: SEED_SNAPSHOT.windowStart,
          windowEnd: SEED_SNAPSHOT.windowEnd,
          generatedLabel: SEED_SNAPSHOT.generatedLabel,
          payload: JSON.stringify(SEED_SNAPSHOT.payload),
        })
        .returning({ id: schema.snapshots.id });
      const inserted = result[0];
      if (!inserted) throw new Error("seedSnapshot: insert returned no rows");
      ctx.invalidateQueries();
      return { seeded: true, id: inserted.id };
    },
  }),
} satisfies ActionsModule;
